"""
bundle.py — Save and load composite documents as directory bundles.

A bundle is a directory containing:
    document.json     — the composite document with large arrays replaced
                        by ``{"$bundle_ref": "arrays/<hash>.parquet"}`` markers
    arrays/           — externalized arrays as Parquet files

This dramatically reduces document size for composites with large numpy
arrays in their configs or state (e.g. sequence arrays, stoichiometry
matrices).  A 2.9 GB JSON document typically compresses to ~50–80 MB.

Usage::

    composite.save_bundle('out/my_bundle')
    loaded = Composite.load_bundle('out/my_bundle', core=core)

The format is designed to be:
- Language-independent (Parquet is readable from Python, R, Rust, Java, …)
- Self-contained (everything needed to reconstruct the composite)
- Human-inspectable (document.json is small enough to open in an editor)
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

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
# Helpers: detect and extract arrays from serialized dicts
# ---------------------------------------------------------------------------

def _estimate_json_size(value) -> int:
    """Rough byte estimate of a value when rendered as JSON."""
    if isinstance(value, (int, float)):
        return 8
    if isinstance(value, str):
        return len(value) + 2
    if isinstance(value, bool) or value is None:
        return 5
    if isinstance(value, list):
        if len(value) == 0:
            return 2
        # Sample first element to estimate
        per_elem = _estimate_json_size(value[0])
        return len(value) * (per_elem + 2)  # +2 for comma+space
    if isinstance(value, dict):
        return sum(
            len(k) + 4 + _estimate_json_size(v)
            for k, v in value.items())
    return 16


def _is_numeric_list(value) -> bool:
    """Check if value is a list of numbers (1D array) or list of lists of
    numbers (2D+ array)."""
    if not isinstance(value, list) or len(value) == 0:
        return False
    first = value[0]
    if isinstance(first, (int, float)):
        return True
    if isinstance(first, list) and len(first) > 0:
        return _is_numeric_list_inner(first)
    return False


def _is_numeric_list_inner(value) -> bool:
    """Recursively check inner lists."""
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list) and len(value) > 0:
        return _is_numeric_list_inner(value[0])
    return False


def _is_structured_array(value) -> bool:
    """Check if value is a vivarium-style __structured_array__ dict."""
    return (isinstance(value, dict)
            and value.get('__structured_array__') is True
            and 'dtype' in value
            and 'data' in value)


def _list_to_ndarray(data, dtype_hint=None):
    """Convert a nested list back to a numpy array.

    Tries to infer the best dtype. For integer data uses int64,
    for mixed int/float uses float64.
    """
    arr = np.array(data)
    # If all values are small ints (0-3 or -1), use int8 (sequence data)
    if arr.dtype.kind in ('i', 'u') and arr.min() >= -1 and arr.max() <= 127:
        arr = arr.astype(np.int8)
    return arr


def _content_hash(data: bytes) -> str:
    """Short content hash for deduplication."""
    return hashlib.sha256(data).hexdigest()[:16]


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

    table = pq.read_table(filepath)
    meta = table.schema.metadata or {}

    # Check for binary blob format (3D+ arrays)
    col_names = table.column_names
    if col_names == ['data', 'shape', 'dtype']:
        shape = json.loads(table.column('shape')[0].as_py())
        dtype = np.dtype(table.column('dtype')[0].as_py())
        data = table.column('data')[0].as_py()
        return np.frombuffer(data, dtype=dtype).reshape(shape)

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
# Extract arrays from a serialized document
# ---------------------------------------------------------------------------

def extract_arrays(
    document: Dict[str, Any],
    arrays_dir: str,
    min_bytes: int = MIN_ARRAY_BYTES,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Walk *document* and replace large arrays with ``$bundle_ref`` markers.

    Returns ``(modified_document, ref_map)`` where *ref_map* maps
    ref paths to filenames.  Arrays are saved to *arrays_dir*.

    Handles:
    - Nested lists of numbers (from numpy array serialization)
    - ``__structured_array__`` dicts (vivarium structured array format)
    """
    os.makedirs(arrays_dir, exist_ok=True)
    ref_map: Dict[str, str] = {}  # content_hash -> filename
    counter = [0]

    def _make_ref(arr: np.ndarray, path_hint: str) -> Dict[str, Any]:
        """Save array and return a $bundle_ref marker."""
        raw = arr.tobytes()
        content_id = _content_hash(raw + str(arr.dtype).encode())

        if content_id in ref_map:
            # Deduplicated — same content already saved
            filename = ref_map[content_id]
        else:
            # Use path hint for a human-readable filename
            safe_hint = path_hint.replace('.', '_').replace('/', '_')
            if len(safe_hint) > 60:
                safe_hint = safe_hint[:60]
            filename = f'{safe_hint}_{content_id}.parquet'
            filepath = os.path.join(arrays_dir, filename)
            _save_array_parquet(arr, filepath)
            ref_map[content_id] = filename

        marker = {
            REF_KEY: f'{ARRAY_DIR}/{filename}',
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
        }
        if arr.dtype.names:
            marker['dtype'] = str(arr.dtype)
            marker['structured'] = True
        return marker

    def _walk(node, path=''):
        """Recursively walk and replace large arrays."""
        if isinstance(node, dict):
            # Check for __structured_array__ format
            if _is_structured_array(node):
                est = _estimate_json_size(node['data'])
                if est >= min_bytes:
                    import ast
                    dtype = np.dtype(ast.literal_eval(node['dtype']))
                    data = node['data']
                    arr = np.array(
                        [tuple(row) for row in data], dtype=dtype)
                    return _make_ref(arr, path)
                return node

            # Walk children
            result = {}
            for key, value in node.items():
                child_path = f'{path}.{key}' if path else key
                result[key] = _walk(value, child_path)
            return result

        if isinstance(node, list):
            # Check if this is a numeric array
            if _is_numeric_list(node):
                est = _estimate_json_size(node)
                if est >= min_bytes:
                    arr = _list_to_ndarray(node)
                    return _make_ref(arr, path)
            else:
                # Walk list elements (might contain dicts with arrays)
                return [_walk(item, f'{path}[{i}]') for i, item in enumerate(node)]

        # Convert numpy scalars and tuples to JSON-native types
        if isinstance(node, (np.integer,)):
            return int(node)
        if isinstance(node, (np.floating,)):
            return float(node)
        if isinstance(node, (np.bool_,)):
            return bool(node)
        if isinstance(node, tuple):
            return [_walk(item, f'{path}[{i}]') for i, item in enumerate(node)]

        return node

    modified = _walk(document)
    return modified, ref_map


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
# Public API: save_bundle / load_bundle
# ---------------------------------------------------------------------------

def save_bundle(
    document: Dict[str, Any],
    outdir: str,
    min_bytes: int = MIN_ARRAY_BYTES,
) -> Dict[str, Any]:
    """Save a composite document as a bundle directory.

    Args:
        document: Serialized composite document (from serialize_state/schema).
        outdir: Bundle directory path (will be created).
        min_bytes: Minimum estimated JSON size to externalize an array.

    Returns:
        Summary dict with file counts and sizes.
    """
    os.makedirs(outdir, exist_ok=True)
    arrays_dir = os.path.join(outdir, ARRAY_DIR)

    # Extract arrays and get modified document
    modified_doc, ref_map = extract_arrays(document, arrays_dir, min_bytes)

    # Write the document
    doc_path = os.path.join(outdir, DOCUMENT_FILE)
    with open(doc_path, 'w') as f:
        json.dump(modified_doc, f, indent=2)

    # Summary
    doc_size = os.path.getsize(doc_path)
    array_sizes = {}
    for filename in ref_map.values():
        fpath = os.path.join(arrays_dir, filename)
        array_sizes[filename] = os.path.getsize(fpath)
    total_array_bytes = sum(array_sizes.values())

    summary = {
        'document_size': doc_size,
        'num_arrays': len(ref_map),
        'total_array_bytes': total_array_bytes,
        'total_bytes': doc_size + total_array_bytes,
        'array_files': array_sizes,
    }

    print(f"Saved bundle to {outdir}/")
    print(f"  document.json: {doc_size / 1e6:.1f} MB")
    print(f"  arrays: {len(ref_map)} files, {total_array_bytes / 1e6:.1f} MB")
    print(f"  total: {(doc_size + total_array_bytes) / 1e6:.1f} MB")

    return summary


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
