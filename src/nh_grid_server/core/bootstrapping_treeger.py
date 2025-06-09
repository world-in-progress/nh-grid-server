from __future__ import annotations
import os
import sys
import time
import yaml
import signal
import logging
import threading
import subprocess
import c_two as cc
from contextlib import contextmanager
from typing import Generator, Type, TypeVar, cast

from ..core.config import settings
from icrms.itreeger import ITreeger, TreeMeta, ReuseAction, SceneNodeInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BSTreeger')

T = TypeVar('T')

class ProxyCRM:
    def __init__(self, icrm_class: Type[T], client: cc.message.Client):
        self._wrapped = icrm_class()
        self._wrapped.client = client
    
    def __getattr__(self, name):
        if hasattr(self._wrapped, name):
            return getattr(self._wrapped, name)
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            logger.error(f'Attribute {name} not found in ProxyCRM or wrapped ICRM {self._wrapped.__class__.__name__}')
            raise AttributeError(f'{name} not found in ProxyCRM or wrapped ICRM {self._wrapped.__class__.__name__}')

    def __del__(self):
        if hasattr(self._wrapped, 'client'):
            try:
                self._wrapped.client.terminate()
            except Exception as e:
                logger.warning(f'Failed to terminate client: {e}')

class BootStrappingTreeger:
    instance: ITreeger | 'BootStrappingTreeger' = None
    _lock = threading.Lock()
    
    def __new__(cls):

        if cls.instance is None:
            with cls._lock:
                if cls.instance is None:
                    cls.instance = super(BootStrappingTreeger, cls).__new__(cls)
                    cls.instance._initialized = False
        return cls.instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        self._process = None
        self._meta_path = settings.SCENARIO_META_PATH
        self._tcp_address = settings.TREEGER_TCP_ADDRESS
        
        if not self._meta_path or not self._tcp_address:
            raise ValueError('Treeger meta path and TCP address must be set in settings')

        try:
            with open(self._meta_path, 'r') as f:
                meta_data = yaml.safe_load(f)
            
            meta = TreeMeta(**(meta_data['meta']))
            for crm_temp in meta.crm_entries:
                if crm_temp.name == 'Treeger':
                    self._crm_launcher = crm_temp.crm_launcher
                    break
            
            # Initialize the CRM process
            self._bootstrap()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f'Failed to initialize Treeger from {self._meta_path}: {e}')
            self._initialized = False
            raise
        
    def _bootstrap(self):
        # Platform-specific subprocess arguments
        kwargs = {}
        if sys.platform != 'win32':
            # Unix-specific: create new process group
            kwargs['preexec_fn'] = os.setsid
        else:
            # Windows-specific: don't open a new console window
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        
        self._process = subprocess.Popen(
            [
                sys.executable,
                self._crm_launcher,
                '--meta_path', self._meta_path,
                '--tcp_address', self._tcp_address,
            ],
            **kwargs
        )
        
        start_time = time.time()
        timeout = 60
        
        logger.info('Waiting for Treeger CRM to start...')
        while True:
            if cc.message.Client.ping(self._tcp_address, timeout=1) is False:
                if time.time() - start_time > timeout:
                    logger.error(f'Timeout waiting for Treeger CRM to start after {timeout} seconds')
                    raise TimeoutError(f'Treeger CRM failed to start within {timeout} seconds')
                logger.info('Waiting for Treeger CRM to start...')
            else:
                logger.info('Treeger CRM started successfully')
                break
    
    def terminate(self):
        # Terminate the CRM process
        if self._process is None:
            return
            
        if sys.platform != 'win32':
            # Unix-specific: terminate the process group
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGINT)
            except (AttributeError, ProcessLookupError):
                self._process.terminate()
        else:
            # Windows-specific: send Ctrl+C signal and then terminate
            try:
                self._process.send_signal(signal.CTRL_C_EVENT)
            except (AttributeError, ProcessLookupError):
                self._process.terminate()

        try:
            self._process.wait(timeout=60)

        except subprocess.TimeoutExpired:
            if sys.platform != 'win32':
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (AttributeError, ProcessLookupError):
                    self._process.kill()
            else:
                self._process.kill()
    
    def __getattr__(self, name):
        icrm = ITreeger()
        if hasattr(icrm, name):
            def method_wrapper(*args, **kwargs):
                with cc.compo.runtime.connect_crm(self._tcp_address, ITreeger) as crm:
                    remote_method = getattr(crm, name)
                    return remote_method(*args, **kwargs)
            return method_wrapper
        else:
            logger.error(f'Attribute {name} not found in ITreeger')
            raise AttributeError(f'{name} not found in ITreeger')
        
    @contextmanager
    def connect(self, node_key: str, icrm: Type[T], deactivate_node_service: bool = False) -> Generator[T, None, None]:
        proxy_crm = None
        try:
            with cc.compo.runtime.connect_crm(self._tcp_address, ITreeger) as crm:
                tcp_address = crm.activate_node(node_key, ReuseAction.FORK)
                
            client = cc.message.Client(tcp_address)
            proxy_crm = ProxyCRM(icrm, client)
            yield proxy_crm
            
        finally:
            try:
                # Terminate the CRM server process
                if deactivate_node_service:
                    with cc.compo.runtime.connect_crm(self._tcp_address, ITreeger) as crm:
                        crm.deactivate_node(node_key)
                    
                # Terminate the client connection
                if proxy_crm and hasattr(proxy_crm, 'client'):
                    proxy_crm.client.terminate()
            except Exception as e:
                logger.warning(f'Failed to disconnect services from node "{node_key}": {e}')

BT = BootStrappingTreeger
