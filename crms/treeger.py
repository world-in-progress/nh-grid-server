import os
import sys
import time
import yaml
import signal
import socket
import logging
import threading
import subprocess
import c_two as cc
from pathlib import Path
from dataclasses import dataclass, field
from icrms.itreeger import ITreeger, CRMEntry, TreeMeta, ReuseAction, ScenarioNode, ScenarioNodeType, SceneNodeInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(os.getcwd()).resolve()

@dataclass
class ProcessInfo():
    port: int
    address: str
    start_time: float = 0.0
    scenario_node_name: str = ''
    process: subprocess.Popen | None = None

@dataclass
class SceneNode():
    node_key: str
    scenario_node: ScenarioNode
    launch_params: dict = field(default_factory=dict)
    
    parent: 'SceneNode' = None
    children: list['SceneNode'] = field(default_factory=list)

    def add_parent(self, parent: 'SceneNode'):
        self.parent = parent
        if parent:
            parent.children.append(self)
    
    def add_child(self, child: 'SceneNode'):
        self.children.append(child)
        child.parent = self
    
    def add_children(self, children: list['SceneNode']):
        for child in children:
            self.add_child(child)

@cc.iicrm
class Treeger(ITreeger):
    def __init__(self, meta_path: str):
        self.meta_path = ROOT_DIR / meta_path
        self.process_pool: dict[str, ProcessInfo] = {}
        self.scene_nodes_in_flight: dict[str, set[str]] = {}  # scenario node name -> set of scene node names

        self._port_lock = threading.Lock()
        
        self.used_ports: set[int] = set()
        
        try:
            with open(meta_path, 'r') as f:
                tree_meta = yaml.safe_load(f)
            self.meta = TreeMeta(**(tree_meta['meta']))
            self.crm_entry_dict: dict[str, CRMEntry] = {
                node.name: node for node in self.meta.crm_entries
            }
            self.max_ports: int = self.meta.configuration.max_ports
            self.port_range: tuple[int, int] = self.meta.configuration.port_range

            # Iterate through the scenario
            self.root = self.meta.scenario
            self.root.parent = None
            self.root.semantic_path = self.root.name
            self.scenario_node_dict: dict[str, ScenarioNode] = {
                self.root.name: self.root
            }
            scenario_node_stack = [self.root]
            while scenario_node_stack:
                # Get the current scenario node
                scenario_node = scenario_node_stack.pop()
                # Record the scenario node in the dictionary
                self.scenario_node_dict[scenario_node.name] = scenario_node
                # Initialize the CRM in-flight set if not exists
                if scenario_node.name not in self.scene_nodes_in_flight:
                    self.scene_nodes_in_flight[scenario_node.name] = set()

                # Parse node type
                if len(scenario_node.children) == 0:
                    scenario_node.node_type = ScenarioNodeType.Resource
                elif len(scenario_node.children) == 1:
                    scenario_node.node_type = ScenarioNodeType.Conception
                else:
                    scenario_node.node_type = ScenarioNodeType.Aggregation
                    
                # Update semantic paths for all children
                for child in scenario_node.children:
                    child.parent = scenario_node
                    child.semantic_path = f'{scenario_node.semantic_path}/{child.name}'
                    scenario_node_stack.append(child)
            
            # Initialize scene
            self.scene: dict[str, SceneNode] = {}
            scene_path = ROOT_DIR / self.meta.configuration.scene_path
            if scene_path.exists():
                logger.info(f'Loading scene from {scene_path}')
                self._deserialize_scene()
                    
            else:
                logger.warning(f'Scene path {scene_path} does not exist, creating a new scene')
                scene_path.parent.mkdir(parents=True, exist_ok=True)
            
                self.scene_node = SceneNode(
                    node_key='root',
                    scenario_node=self.root,
                    launch_params={
                        'meta_path': meta_path,
                    }
                )
                self.scene['root'] = self.scene_node
            
        except Exception as e:
            logger.error(f'Failed to initialize treeger from {meta_path}: {e}')

    def mount_node(self, scenario_node_name: str, node_key: str, launch_params: dict | None = None, start_service_immediately: bool = False, reusibility: ReuseAction = ReuseAction.FORK) -> bool:
        if node_key in self.scene:
            logger.warning(f'Node {node_key} already mounted, skipping')
            return True
        
        scenario_node = self.scenario_node_dict.get(scenario_node_name, None)
        if scenario_node is None:
            logger.error(f'Scenario node {scenario_node_name} not found in tree meta')
            raise ValueError(f'Scenario node {scenario_node_name} not found in tree meta')
        
        if not scenario_node.crm and launch_params is not None:
            logger.warning(f'Launch parameters provided for node "{scenario_node_name}" not having a CRM, ignoring launch_params {launch_params}')
            launch_params = {}
        
        # Validate node_key
        parent_key = '/'.join(node_key.split('/')[:-1])
        parent_node = self.scene.get(parent_key, None)
        if not parent_node:
            raise ValueError(f'Parent node "{parent_key}" not found in scene for node "{node_key}"')

        # Create the SceneNode instance
        node = SceneNode(
            node_key=node_key,
            scenario_node=scenario_node,
            launch_params=launch_params
        )
        parent_node.add_child(node)
        
        # Add node to the scene
        self.scene[node_key] = node
        logger.info(f'Successfully mounted node "{node_key}" for scenario "{scenario_node_name}"')

        # If the node should start immediately, activate it
        if start_service_immediately:
            try:
                self.activate_node(node_key, reusibility)
            except Exception as e:
                logger.error(f'Failed to activate node "{node_key}": {e}')
                return False
        
        return True
    
    def _unmount_node_recursively(self, node_key: str) -> bool:
        if node_key not in self.scene:
            logger.warning(f'Node "{node_key}" not found in scene, cannot unmount')
            return False
        
        # Recursively iterate through the children and unmount them
        for child in self.scene[node_key].children:
            self._unmount_node_recursively(child.node_key)
        
        # Stop the node service if it's running
        if node_key in self.process_pool:
            self.deactivate_node(node_key)
        
        # Remove the node from the scene
        node = self.scene[node_key]
        if node.parent:
            node.parent.children.remove(node)
        
        del self.scene[node_key]
        logger.info(f'Successfully unmounted node {node_key}')
        return True

    def unmount_node(self, node_key: str) -> bool:
        return self._unmount_node_recursively(node_key)

    def _serialize_scene(self) -> str:
        scene_data = []
        for _, scene_node in self.scene.items():
            scene_data.append({
                'node_key': scene_node.node_key,
                'scenario_node_name': scene_node.scenario_node.name,
                'launch_params': scene_node.launch_params,
                'parent_key': scene_node.parent.node_key if scene_node.parent else None
            })
        
        scene_path = ROOT_DIR / self.meta.configuration.scene_path
        with open(scene_path, 'w') as f:
            yaml.dump(scene_data, f, default_flow_style=False)
        logger.info(f'Scene serialized to {scene_path}')
    
    def _deserialize_scene(self) -> None:
        scene_path = ROOT_DIR / self.meta.configuration.scene_path
        if not scene_path.exists():
            logger.warning(f'Scene file {scene_path} does not exist, skipping deserialization')
            return
        
        self.scene.clear()
        with open(scene_path, 'r') as f:
            scene_data = yaml.safe_load(f)
            for scene_node_data in scene_data:
                scene_node = SceneNode(
                    node_key=scene_node_data['node_key'],
                    scenario_node=self.scenario_node_dict.get(scene_node_data['scenario_node_name']),
                    launch_params=scene_node_data['launch_params'],
                )
                self.scene[scene_node.node_key] = scene_node
            
            for scene_node_data in scene_data:
                node = self.scene[scene_node_data['node_key']]
                if scene_node_data['parent_key'] and scene_node_data['parent_key'] in self.scene:
                    parent_node = self.scene[scene_node_data['parent_key']]
                    node.add_parent(parent_node)

    def terminate(self) -> bool:
        try:
            for node_key in list(self.process_pool.keys()):
                self.deactivate_node(node_key)
            
            logger.info('All nodes stopped successfully')
            
            self._serialize_scene()
            
            return True
        except Exception as e:
            logger.error(f'Failed to terminate treeger: {e}')
            return False
            
    def _is_port_available(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                sock.listen(1)
            
            return True
        
        except (OSError, socket.error):
            return False
    
    def _find_available_port(self) -> int | None:
        start_port, end_port = self.port_range
        for port in range(start_port, end_port + 1):
            if port not in self.used_ports and self._is_port_available(port):
                return port
        
        for port in range(start_port, end_port + 1):
            if self._is_port_available(port):
                self.used_ports.discard(port)
                return port
    
    def _release_node_port(self, node_key: str):
        if node_key in self.process_pool:
            process_info = self.process_pool[node_key]
            if process_info.port:
                self.used_ports.discard(process_info.port)
                logger.info(f'Released port {process_info.port} for node {node_key}')
            
            # Remove record from process pool and scene node in-flight set
            del self.process_pool[node_key]
            self.scene_nodes_in_flight[process_info.scenario_node_name].remove(node_key)
    
    def _cleanup_finished_processes(self):
        finished_nodes = []
        
        for node_name, node_info in self.process_pool.items():
            process = node_info.process
            if process and process.poll() is not None:
                finished_nodes.append(node_name)
        
        for node_name in finished_nodes:
            self._release_node_port(node_name)
    
    def _wait_for_available_slot(self, time_out: float = 30.0) -> bool:
        start_time = time.time()
        
        while len(self.used_ports) >= self.max_ports:
            if time.time() - start_time > time_out:
                logger.error('Timeout waiting for available slot')
                return False
            
            logger.info(f'Process pool full ({len(self.used_ports)}/{self.max_ports}), waiting...')
            time.sleep(1.0)
            
            self._cleanup_finished_processes()
        
        return True
    
    def _try_get_address(self) -> str:
        with self._port_lock:
            if not self._wait_for_available_slot():
                raise RuntimeError('Unable to allocate port: process pool full and timeout reached')
            
            port = self._find_available_port()
            if port is None:
                raise RuntimeError(f'No available ports in range {self.port_range}')
            
            # TODO: Should use reserved ports?
            
            self.used_ports.add(port)
            address = f'tcp://127.0.0.1:{port}'
            
            logger.debug(f'Allocated address {address}')
            return address
    
    def activate_node(self, node_key: str, reusibility: ReuseAction = ReuseAction.REPLACE) -> str:
        # Check if the node is valid
        node = self.scene.get(node_key)
        if not node:
            raise ValueError(f'Node {node_key} not found in scene')
        
        # Check if the node can be launched
        if not node.scenario_node.crm:
            raise ValueError(f'Node {node_key} does not have a CRM and cannot be launched directly')

        # Check if the node is already running
        if node_key in self.process_pool:
            process_info = self.process_pool[node_key]
            return process_info.address
        
        # Handle reusability actions
        flying_sibling_set = self.scene_nodes_in_flight.get(node.scenario_node.name)
        # Get the first available node sharing the same scenario node
        sibling_node_name = next(iter(flying_sibling_set), None)
        if sibling_node_name:
            if reusibility == ReuseAction.KEEP:
                # Keep the crm process
                sibling_process_info = self.process_pool.get(sibling_node_name)
                return sibling_process_info.address

            elif reusibility == ReuseAction.REPLACE:
                # Replace the sibling node with the new one (stop the sibling process and create below)
                self.deactivate_node(sibling_node_name)

            elif reusibility == ReuseAction.FORK:
                # Fork the sibling node, which means creating a new process for the node but keeping the sibling process running
                pass

        # Try to allocate an address for the node
        try:
            address = self._try_get_address()
            port = int(address.split(':')[-1])
        except Exception as e:
            logger.error(f'Failed to allocate address for node {node_key}: {e}')
            raise

        # Try to launch a CRM server related to the node
        try:
            # Platform-specific subprocess arguments
            kwargs = {}
            if sys.platform != 'win32':
                # Unix-specific: create new process group
                kwargs['preexec_fn'] = os.setsid
            
            # Assmble the command to launch the CRM server
            params = node.launch_params
            crm_entry: CRMEntry = self.crm_entry_dict.get(node.scenario_node.crm, None)
            if crm_entry is None:
                raise ValueError(f'CRM template {node.scenario_node.crm} not found in tree meta')
            
            cmd = [
                sys.executable,
                crm_entry.crm_launcher,
                '--tcp_address', address,
            ]
            for key, value in params.items():
                cmd.extend([f'--{key}', str(value)])
            
            process = subprocess.Popen(
                cmd,
                **kwargs
            )
            
            # Register the process in the process pool and scene node in-flight set
            self.process_pool[node_key] = ProcessInfo(
                port=port,
                address=address,
                process=process,
                start_time=time.time(),
                scenario_node_name=node.scenario_node.name
            )
            self.scene_nodes_in_flight[node.scenario_node.name].add(node_key)

            logger.info(f'Successfully launched node "{node_key}" at {address}')
            return address

        except Exception as e:
            if 'port' in locals():
                self.used_ports.discard(port)
            logger.error(f'Failed to launch node {node_key}: {e}')
            raise

    def deactivate_node(self, node_key: str) -> bool:
        if node_key not in self.process_pool:
            logger.warning(f'Node "{node_key}" not found in process pool')
            return False
        
        try:
            process_info = self.process_pool[node_key]
            process = process_info.process
            tcp_address = process_info.address
            if cc.message.Client.shutdown(tcp_address, timeout=60) is False:
                raise RuntimeError(f'Failed to shutdown node "{node_key}" at {tcp_address}')
                
            self._release_node_port(node_key)
            
            logger.info(f'Successfully stopped node "{node_key}"')
            return True
        
        except Exception as e:
            logger.error(f'Failed to stop node "{node_key}": {e}')
            return False
    
    def get_node_info(self, node_key: str) -> SceneNodeInfo | None:
        # Check if the node exists in the scene
        if node_key not in self.scene:
            logger.warning(f'Node "{node_key}" not found in scene')
            return None
        
        # Get the SceneNode instance
        scene_node = self.scene[node_key]
        
        # Get the TCP address of the node if it is running
        if node_key in self.process_pool:
            process_info = self.process_pool[node_key]
            if process_info.process and process_info.process.poll() is None:
                # Process is running, return its address
                tcp_address = process_info.address
            else:
                # Process is not running, return None
                tcp_address = None
                # Cleanup the process pool entry
                self._release_node_port(node_key)
        
        # Prepare the node info
        node_info = SceneNodeInfo(
            node_key=scene_node.node_key,
            scenario_node_name=scene_node.scenario_node.name,
            parent_key=scene_node.parent.node_key if scene_node.parent else None,
            tcp_address=tcp_address
        )
        return node_info

    def get_process_pool_status(self) -> dict:
        self._cleanup_finished_processes()
        
        running_nodes = []
        for node_name, node_info in self.process_pool.items():
            process = node_info.process
            status = 'running' if process and process.poll() is None else 'stopped'
            running_nodes.append({
                'status': status,
                'name': node_name,
                'address': node_info.address,
                'template': node_info.scenario_node_name,
                'uptime': time.time() - node_info.start_time
            })
        
        return {
            'used_ports': len(self.used_ports),
            'max_ports': self.max_ports,
            'available_slots': self.max_ports - len(self.used_ports),
            'nodes': running_nodes,
            'port_range': self.port_range
        }