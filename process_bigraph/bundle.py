"""
bundle.py — Array Parquet I/O and bundle loading for composite documents.

A bundle is a directory containing:
    document.json     — the composite document with large arrays replaced
                        by ``{"$bundle_ref": "arrays/<hash>.parquet"}`` markers
    arrays/           — externalized arrays as Parquet files

This module provides the array externalization primitives
(``_save_array_parquet`` / ``_load_array_parquet``) used by the typed-dispatch
``bundle`` method in bigraph-schema, plus the bundle *load* path
(``resolve_refs`` / ``load_bundle``). The *save* path lives on
``Composite.save_bundle``, which drives ``core.bundle`` + ``BundleContext``.

The format is designed to be:
- Language-independent (Parquet is readable from Python, R, Rust, Java, …)
- Self-contained (everything needed to reconstruct the composite)
- Human-inspectable (document.json is small enough to open in an editor)
"""

import json
import os
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARRAY_DIR = 'arrays'
DOCUMENT_FILE = 'document.json'
REF_KEY = '$bundle_ref'

# Minimum estimated JSON size (bytes) before an array gets externalized.
# 10 KB — small arrays stay inline for readability.
MIN_ARRAY_BYTES = 10_000


# ---------------------------------------------------------------------------
# Parquet I/O for arrays
# ---------------------------------------------------------------------------

def _save_array_parquet(arr: np.ndarray, filepath: str) -> None:
    """Save a numpy array to a Parquet file.

    Strategy:
    - 1D/2D numeric arrays: store as columnar Parquet (one column per
      array column, or a single column for 1D).
    - Higher-dimensional or structured arrays: store as a binary blob
      with shape/dtype metadata.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if arr.dtype.names:
        # Structured array — each field becomes one or more Parquet columns.
        # Sub-array fields (e.g. shape (N, 2)) are flattened into separate
        # columns: field.0, field.1, etc.
        columns = {}
        subarray_meta = {}  # field_name -> {'shape': [...], 'dtype': '...'}

        for name in arr.dtype.names:
            field_data = arr[name]
            if field_data.dtype.kind == 'U':
                columns[name] = pa.array(field_data.tolist(), type=pa.string())
            elif field_data.ndim > 1:
                # Sub-array field: flatten into N columns
                sub_shape = field_data.shape[1:]
                sub_count = int(np.prod(sub_shape))
                flat = field_data.reshape(len(field_data), sub_count)
                for i in range(sub_count):
                    columns[f'{name}.{i}'] = pa.array(flat[:, i])
                subarray_meta[name] = {
                    'shape': list(sub_shape),
                    'dtype': str(field_data.dtype.base),
                    'count': sub_count,
                }
            else:
                columns[name] = pa.array(field_data)

        table = pa.table(columns)

        # Store the original dtype string and sub-array info in file metadata
        file_meta = {
            'dtype': str(arr.dtype),
            'subarray_fields': subarray_meta,
        }
        schema_meta = table.schema.metadata or {}
        schema_meta[b'bundle_structured'] = json.dumps(file_meta).encode()
        table = table.replace_schema_metadata(schema_meta)

        pq.write_table(table, filepath, compression='zstd')

    elif arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] <= 500):
        # Dense 1D or moderate 2D array — columnar storage.
        if arr.ndim == 1:
            table = pa.table({'c0': pa.array(arr)})
        else:
            columns = {
                f'c{i}': pa.array(arr[:, i])
                for i in range(arr.shape[1])
            }
            table = pa.table(columns)

        # Store original dtype in metadata
        meta = {b'dtype': str(arr.dtype).encode(),
                b'shape': json.dumps(list(arr.shape)).encode()}
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, filepath, compression='zstd')

    else:
        # Wide 2D (>500 cols) or 3D+ — binary blob with compression.
        # Storing as raw bytes in a single Parquet row with zstd
        # compression. This is still very efficient: a 50MB int8 array
        # compresses to ~15MB.
        table = pa.table({
            'data': [arr.tobytes()],
            'shape': [json.dumps(list(arr.shape))],
            'dtype': [str(arr.dtype)],
        })
        pq.write_table(table, filepath, compression='zstd')


def _load_array_parquet(filepath: str) -> np.ndarray:
    """Load a numpy array from a Parquet file created by _save_array_parquet."""
    import pyarrow.parquet as pq

    # Use ParquetFile.read() rather than pq.read_table: the latter
    # auto-infers hive partition columns from the path (e.g.
    # ``variant=0/seed=1/...``) and appends them to every table read,
    # which both clobbers metadata and adds spurious data columns.
    table = pq.ParquetFile(filepath).read()
    meta = table.schema.metadata or {}

    # Check for binary blob format (3D+ arrays)
    col_names = table.column_names
    if col_names == ['data', 'shape', 'dtype']:
        shape = json.loads(table.column('shape')[0].as_py())
        dtype = np.dtype(table.column('dtype')[0].as_py())
        data = table.column('data')[0].as_py()
        # np.frombuffer returns a read-only view; copy to make writable
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    # Check for structured array
    if b'bundle_structured' in meta:
        import ast
        struct_meta = json.loads(meta[b'bundle_structured'].decode())
        dtype = np.dtype(ast.literal_eval(struct_meta['dtype']))
        subarray_fields = struct_meta.get('subarray_fields', {})

        n_rows = table.num_rows
        result = np.zeros(n_rows, dtype=dtype)

        for name in dtype.names:
            if name in subarray_fields:
                # Reconstruct sub-array from flattened columns
                info = subarray_fields[name]
                sub_count = info['count']
                sub_shape = info['shape']
                cols = [table.column(f'{name}.{i}').to_numpy()
                        for i in range(sub_count)]
                flat = np.column_stack(cols)
                result[name] = flat.reshape((n_rows,) + tuple(sub_shape))
            elif dtype[name].kind == 'U':
                result[name] = table.column(name).to_pylist()
            else:
                result[name] = table.column(name).to_numpy()

        return result

    # Dense numeric array
    dtype_str = meta.get(b'dtype', b'float64').decode()
    shape_str = meta.get(b'shape', b'null').decode()
    dtype = np.dtype(dtype_str)

    if len(col_names) == 1:
        arr = table.column('c0').to_numpy().astype(dtype)
    else:
        cols = [table.column(f'c{i}').to_numpy() for i in range(len(col_names))]
        arr = np.column_stack(cols).astype(dtype)

    if shape_str != 'null':
        shape = json.loads(shape_str)
        arr = arr.reshape(shape)

    return arr


# ---------------------------------------------------------------------------
# Resolve $bundle_ref markers back into data
# ---------------------------------------------------------------------------

def resolve_refs(
    document: Dict[str, Any],
    bundle_dir: str,
    as_numpy: bool = False,
) -> Dict[str, Any]:
    """Walk *document* and replace ``$bundle_ref`` markers with loaded data.

    Args:
        document: The document dict (from document.json).
        bundle_dir: Path to the bundle directory.
        as_numpy: If True, return numpy arrays directly. If False (default),
            return nested Python lists (for passing to Composite constructor
            which handles realize).
    """

    def _walk(node):
        if isinstance(node, dict):
            if REF_KEY in node:
                ref_path = os.path.join(bundle_dir, node[REF_KEY])
                arr = _load_array_parquet(ref_path)
                if node.get('structured'):
                    # Reconstruct __structured_array__ format for
                    # downstream deserializers that expect it
                    if as_numpy:
                        return arr
                    return {
                        '__structured_array__': True,
                        'dtype': str(arr.dtype),
                        'data': [list(row) for row in arr.tolist()],
                    }
                if as_numpy:
                    return arr
                return arr.tolist()
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(document)


# ---------------------------------------------------------------------------
# Public API: load_bundle (the save path lives on Composite.save_bundle)
# ---------------------------------------------------------------------------

def load_bundle(
    bundle_dir: str,
    as_numpy: bool = False,
) -> Dict[str, Any]:
    """Load a composite document from a bundle directory.

    Args:
        bundle_dir: Path to the bundle directory.
        as_numpy: If True, arrays are returned as numpy arrays.
            If False, as nested Python lists (for Composite constructor).

    Returns:
        The fully resolved document dict.
    """
    doc_path = os.path.join(bundle_dir, DOCUMENT_FILE)
    with open(doc_path, 'r') as f:
        document = json.load(f)

    return resolve_refs(document, bundle_dir, as_numpy=as_numpy)
