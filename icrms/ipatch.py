import c_two as cc
import pyarrow as pa

# Define transferables ##################################################

@cc.transferable
class GridSchema:
    """
    Grid Schema
    ---
    - epsg (int): the EPSG code of the grid
    - bounds (list[float]): the bounds of the grid in the format [min_x, min_y, max_x, max_y]
    - first_size (float): the size of the first grid (unit: m)
    - subdivide_rules (list[tuple[int, int]]): the subdivision rules of the grid in the format [(sub_width, sub_height)]
    """
    epsg: int
    bounds: list[float]  # [min_x, min_y, max_x, max_y]
    first_size: list[float] # [width, height]
    subdivide_rules: list[list[int]]  # [(sub_width, sub_height), ...]
        
    def serialize(grid_schema: 'GridSchema') -> bytes:
        arrow_schema = pa.schema([
            pa.field('epsg', pa.int32()),
            pa.field('bounds', pa.list_(pa.float64())),
            pa.field('first_size', pa.list_(pa.float64())),
            pa.field('subdivide_rules', pa.list_(pa.list_(pa.int32())))
        ])
        
        data = {
            'epsg': grid_schema.epsg,
            'bounds': grid_schema.bounds,
            'first_size': grid_schema.first_size,
            'subdivide_rules': grid_schema.subdivide_rules
        }
        
        table = pa.Table.from_pylist([data], schema=arrow_schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> 'GridSchema':
        row = deserialize_to_rows(arrow_bytes)[0]
        return GridSchema(
            epsg=row['epsg'],
            bounds=row['bounds'],
            first_size=row['first_size'],
            subdivide_rules=row['subdivide_rules']
        )

@cc.transferable
class GridAttribute:
    """
    Attributes of Grid
    ---
    - level (uint8): the level of the grid
    - type (uint8): the type of the grid, default to 0
    - activate (bool), the subdivision status of the grid
    - deleted (bool): the deletion status of the grid, default to False
    - elevation (float64): the elevation of the grid, default to -9999.0
    - global_id (uint32): the global id within the bounding box that subdivided by grids all in the level of this grid
    - local_id (uint32): the local id within the parent grid that subdivided by child grids all in the level of this grid
    - min_x (float64): the min x coordinate of the grid
    - min_y (float64): the min y coordinate of the grid
    - max_x (float64): the max x coordinate of the grid
    - max_y (float64): the max y coordinate of the grid
    """
    level: int
    type: int
    activate: bool
    global_id: int
    deleted: bool = False   
    elevation: float = -9999.0
    local_id: int | None = None
    min_x: float | None = None
    min_y: float | None = None
    max_x: float | None = None
    max_y: float | None = None
    
    def serialize(data: 'GridAttribute') -> bytes:
        schema = pa.schema([
            pa.field('deleted', pa.bool_()),
            pa.field('activate', pa.bool_()),
            pa.field('type', pa.uint8()),
            pa.field('level', pa.uint8()),
            pa.field('global_id', pa.uint32()),
            pa.field('local_id', pa.uint32(), nullable=True),
            pa.field('elevation', pa.float64()),
            pa.field('min_x', pa.float64(), nullable=True),
            pa.field('min_y', pa.float64(), nullable=True),
            pa.field('max_x', pa.float64(), nullable=True),
            pa.field('max_y', pa.float64(), nullable=True),
        ])
        
        table = pa.Table.from_pylist([data.__dict__], schema=schema)
        return serialize_from_table(table)
    
    def deserialize(arrow_bytes: bytes) -> 'GridAttribute':
        row = deserialize_to_rows(arrow_bytes)[0]
        return GridAttribute(
            deleted=row['deleted'],
            activate=row['activate'],
            type=row['type'],
            level=row['level'],
            global_id=row['global_id'],
            local_id=row['local_id'],
            elevation=row['elevation'],
            min_x=row['min_x'],
            min_y=row['min_y'],
            max_x=row['max_x'],
            max_y=row['max_y']
        )

@cc.transferable
class GridInfo:
    def serialize(level: int, global_id: int) -> bytes:
        schema = pa.schema([
            pa.field('level', pa.uint8()),
            pa.field('global_id', pa.uint32())
        ])
        
        data = {
            'level': level,
            'global_id': global_id
        }
        
        table = pa.Table.from_pylist([data], schema=schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> tuple[int, int]:
        row = deserialize_to_rows(arrow_bytes)[0]
        return (
            row['level'],
            row['global_id']
        )

@cc.transferable
class PeerGridInfos:
    def serialize(level: int, global_ids: list[int]) -> bytes:
        schema = pa.schema([
            pa.field('level', pa.uint8()),
            pa.field('global_ids', pa.list_(pa.uint32()))
        ])
        
        data = {
            'level': level,
            'global_ids': global_ids
        }
        
        table = pa.Table.from_pylist([data], schema=schema)
        return serialize_from_table(table)

    def deserialize(bytes: bytes) -> tuple[int, list[int]]:
        row = deserialize_to_rows(bytes)[0]
        return (
            row['level'],
            row['global_ids']
        )

@cc.transferable
class GridInfos:
    def serialize(levels: list[int], global_ids: list[int]) -> bytes:
        schema = pa.schema([
            pa.field('levels', pa.uint8()),
            pa.field('global_ids', pa.uint32())
        ])
        table = pa.Table.from_arrays(
            [
                pa.array(levels, type=pa.uint8()), 
                pa.array(global_ids, type=pa.uint32())
            ],
            schema=schema
        )
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> tuple[list[int], list[list[int]]]:
        table = deserialize_to_table(arrow_bytes)
        levels = table.column('levels').to_pylist()
        global_ids = table.column('global_ids').to_pylist()
        return levels, global_ids

@cc.transferable
class GridAttributes:
    def serialize(data: list[GridAttribute]) -> bytes:
        schema = pa.schema([
            pa.field('attribute_bytes', pa.list_(pa.binary())),
        ])

        data_dict = {
            'attribute_bytes': [GridAttribute.serialize(grid) for grid in data]
        }
        
        table = pa.Table.from_pylist([data_dict], schema=schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> list[GridAttribute]:
        table = deserialize_to_table(arrow_bytes)
        
        grid_bytes = table.column('attribute_bytes').to_pylist()[0]
        
        return [GridAttribute.deserialize(grid_byte) for grid_byte in grid_bytes]

@cc.transferable
class GridKeys:
    def serialize(keys: list[str | None]) -> bytes:
        schema = pa.schema([pa.field('keys', pa.string())])
        data = {'keys': keys}
        table = pa.Table.from_pydict(data, schema=schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> list[str | None]:
        table = deserialize_to_table(arrow_bytes)
        keys = table.column('keys').to_pylist()
        return keys

@cc.transferable
class GridCenter:
    def serialize(lon: float, lat: float) -> bytes:
        schema = pa.schema([
            pa.field('lon', pa.float64()),
            pa.field('lat', pa.float64()),
        ])
        
        data = {
            'lon': lon,
            'lat': lat,
        }
        
        table = pa.Table.from_pylist([data], schema=schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> tuple[float, float]:
        row = deserialize_to_rows(arrow_bytes)[0]
        return (
            row['lon'],
            row['lat']
        )

@cc.transferable
class MultiGridCenters:
    def serialize(centers: list[tuple[float, float]]) -> bytes:
        schema = pa.schema([
            pa.field('lon', pa.float64()),
            pa.field('lat', pa.float64()),
        ])
        
        data = {
            'lon': [center[0] for center in centers],
            'lat': [center[1] for center in centers],
        }
        
        table = pa.Table.from_pydict(data, schema=schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> list[tuple[float, float]]:
        table = deserialize_to_table(arrow_bytes)
        lon = table.column('lon').to_pylist()
        lat = table.column('lat').to_pylist()
        return list(zip(lon, lat))

@cc.transferable
class FloatArray:
    def serialize(data: list[float]) -> bytes:
        schema = pa.schema([
            pa.field('data', pa.float64())
        ])
        data = {'data': data}
        table = pa.Table.from_pydict(data, schema=schema)
        return serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> list[float]:
        table = deserialize_to_table(arrow_bytes)
        data = table.column('data').to_pylist()
        return data

@cc.transferable
class TopoSaveInfo:
    success: bool
    message: str
    
    def serialize(info: 'TopoSaveInfo') -> bytes:
        schema = pa.schema([
            pa.field('success', pa.bool_()),
            pa.field('message', pa.string()),
        ])
        
        table = pa.Table.from_pylist([info.__dict__], schema=schema)
        return serialize_from_table(table)
    
    def deserialize(arrow_bytes: bytes) -> 'TopoSaveInfo':
        row = deserialize_to_rows(arrow_bytes)[0]
        return TopoSaveInfo(
            success=row['success'],
            message=row['message'],
        )

# Define ICRM ###########################################################

@cc.icrm
class IPatch:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def get_schema(self) -> GridSchema:
        ...
    
    def get_local_id(self, level: int, global_id: int) -> int:
        ...
    
    def subdivide_grids(self, levels: list[int], global_ids: list[int]) -> tuple[list[int], list[int]]:
        ...
        
    def delete_grids(self, levels: list[int], global_ids: list[int]):
        ...
    
    def get_parents(self, levels: list[int], global_ids: list[int]) -> tuple[list[int], list[int]]:
        ...
        
    def get_status(self, index: int) -> int:
        ...
    
    def get_active_grid_infos(self) -> tuple[list[int], list[int]]:
        ...
    
    def get_deleted_grid_infos(self) -> tuple[list[int], list[int]]:
        ...
    
    def get_multi_grid_bboxes(self, levels: list[int], global_ids: list[int]) -> list[float]:
        ...
        
    def merge_multi_grids(self, levels: list[int], global_ids: list[int]) -> tuple[list[int], list[int]]:
        ...
        
    def recover_multi_grids(self, levels: list[int], global_ids: list[int]):
        ...
        
    def save(self) -> TopoSaveInfo:
        ...

# Helpers ##################################################

def serialize_from_table(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    binary_data = sink.getvalue().to_pybytes()
    return binary_data

def deserialize_to_table(serialized_data: bytes) -> pa.Table:
    buffer = pa.py_buffer(serialized_data)
    with pa.ipc.open_stream(buffer) as reader:
        table = reader.read_all()
    return table

def deserialize_to_rows(serialized_data: bytes) -> dict:
    buffer = pa.py_buffer(serialized_data)

    with pa.ipc.open_stream(buffer) as reader:
        table = reader.read_all()

    return table.to_pylist()