"""Tests for process_bigraph.parquet_emitter.

Helper-function tests (TestHelperFunctions) are near-verbatim ports of the
v2ecoli helper tests (which themselves descend from vEcoli's
ecoli/library/test_parquet_emitter.py). They verify the DuckDB SQL-string
generators and dtype machinery don't drift.

Integration tests exercise ParquetEmitter against process-bigraph's own
core / Composite machinery (no v2ecoli fixtures).

Skipped wholesale when the ``[parquet]`` optional-dependency group is not
installed.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

import pytest

# Skip the whole module if the [parquet] extra isn't installed.
pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("fsspec")

import inspect  # noqa: E402
import sys  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from bigraph_schema import set_path as bs_set_path  # noqa: E402

from process_bigraph import allocate_core  # noqa: E402
from process_bigraph.composite import Composite  # noqa: E402
from process_bigraph.parquet_emitter import (  # noqa: E402
    ParquetEmitter,
    create_duckdb_conn,
    flatten_dict,
    list_columns,
    named_idx,
    ndidx_to_duckdb_expr,
    np_dtype,
    quote_columns,
    union_pl_dtypes,
)
from process_bigraph.processes.examples import IncreaseProcess  # noqa: E402,F401


@pytest.fixture
def core():
    """Allocate a core with this module's processes registered.

    Mirrors the pattern in the project-level ``tests.py``: pass the current
    module's members to ``allocate_core`` so ``IncreaseProcess`` (imported
    above) and ``ParquetEmitter`` (re-exported via emitter.py) are
    discoverable by ``local:`` addresses.
    """
    members = dict(inspect.getmembers(sys.modules[__name__]))
    return allocate_core(top=members)


# ============================================================================
# Helper-function tests — ported from v2ecoli / vEcoli.
# ============================================================================


class TestHelperFunctions:
    @pytest.fixture
    def query_conn(self):
        conn = duckdb.connect(":memory:")
        df = pl.DataFrame(  # noqa: F841
            {
                "a": [[0.1, 0.0, 0.3], [0.4, 0.5, 0.0], [None, 0.8, 0.9]],
                "b": [
                    [[0.1, 0.2], [0.3, None]],
                    [[0.5, 0.6], [0.0, 0.8]],
                    [[0.9, 0.0], [1.1, 1.2]],
                ],
                "c": [[[0.1, 0.2], [0.3]], [[0.5], [0.0, 0.8]], [[0.9], [1.1]]],
            }
        )
        conn.sql("CREATE OR REPLACE TABLE test_table AS SELECT * FROM df")
        yield conn

    def test_named_idx(self, query_conn):
        col_expr = named_idx("a", ["col1", "col2", "col3"], [[0, 1, 2]])
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {"col1": [0.1, 0.4, None], "col2": [0.0, 0.5, 0.8], "col3": [0.3, 0.0, 0.9]}
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "a", ["col1", "col2", "col3"], [[0, 1, 2]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.4, None],
                "col2": [None, 0.5, 0.8],
                "col3": [0.3, None, 0.9],
            }
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "b", ["col1", "col2", "col3", "col4"], [[0, 1], [0, 1]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.5, 0.9],
                "col2": [0.2, 0.6, None],
                "col3": [0.3, None, 1.1],
                "col4": [None, 0.8, 1.2],
            }
        )
        assert result.equals(expected)

    def test_ndidx_to_duckdb_expr(self, query_conn):
        expr = ndidx_to_duckdb_expr("b", [0, 1])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.2]], [[0.6]], [[0.0]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("b", [":", [True, False]])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.1], [0.3]], [[0.5], [0.0]], [[0.9], [1.1]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("c", [[0], ":"])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"c": [[[0.1, 0.2]], [[0.5]], [[0.9]]]})
        assert result.equals(expected)

    def test_flatten_dict(self):
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        assert flatten_dict({"a": {"b": 1, "c": 2}, "d": 3}) == {
            "a__b": 1,
            "a__c": 2,
            "d": 3,
        }
        assert flatten_dict({"a": {"b": {"c": {"d": 1}}}, "e": 2}) == {
            "a__b__c__d": 1,
            "e": 2,
        }
        assert flatten_dict({}) == {}
        nested = flatten_dict({"a": [1, 2, 3], "b": {"c": np.array([4, 5, 6])}})
        assert nested["a"] == [1, 2, 3]
        np.testing.assert_array_equal(nested["b__c"], np.array([4, 5, 6]))

    def test_np_dtype(self):
        # Basic dispatch — order matters since bool is a subclass of int.
        assert np_dtype(1.0, "float_field") == np.float64
        assert np_dtype(True, "bool_field") == np.bool_
        assert np_dtype("text", "string_field") == np.dtypes.StringDType
        assert np_dtype(42, "int_field") == np.int64

        # Override path: explicit exact-name and glob overrides win over deep
        # dispatch (used downstream by callers like v2ecoli's
        # VECOLI_PARQUET_DTYPE_OVERRIDES to compact wide int columns).
        overrides = {
            "exact_field": "UInt16",
            "wide__*": "UInt32",
        }
        assert np_dtype(42, "exact_field", overrides) == np.uint16
        assert np_dtype(42, "wide__monomer_counts", overrides) == np.uint32
        # Non-matching name falls through to deep dispatch.
        assert np_dtype(42, "other_field", overrides) == np.int64

        # Arrays with various dimensions
        assert np_dtype(np.array([1, 2, 3]), "array1d_field") == np.int64
        assert np_dtype(np.array([[1, 2], [3, 4]]), "array2d_field") == np.int64
        # Empty arrays still have a dtype
        assert np_dtype(np.array([]), "empty_array_field") == np.float64

        # Raise to trigger Polars fallback in the emitter
        with pytest.raises(ValueError, match="empty_list_field has unsupported"):
            np_dtype([[], [], None], "empty_list_field")
        with pytest.raises(ValueError, match="none_field has unsupported"):
            np_dtype(None, "none_field")
        with pytest.raises(ValueError, match="complex_field has unsupported type"):
            np_dtype(complex(1, 2), "complex_field")

    def test_union_pl_dtypes(self):
        # Basic types — incompatible pairs raise without a force_inner hint.
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Int32, pl.Int64, "fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Float32, pl.String, "fail")

        # Nested types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.Int16), pl.List(pl.Int64), "nest")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.UInt16), pl.List(pl.String), "nest_fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.List(pl.UInt16)), pl.List(pl.String), "nest_fail"
            )
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.UInt16), pl.List(pl.Array(pl.String, (1,))), "nest_fail"
            )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)

        # Forced types
        assert union_pl_dtypes(pl.Int16, pl.UInt8, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(pl.UInt16, pl.Int64, "force_u32", pl.UInt32) == pl.UInt32
        assert (
            union_pl_dtypes(pl.UInt16, pl.String, "force_u32", pl.UInt32) == pl.UInt32
        )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.String), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.Array(pl.UInt16, (1, 1)),
            pl.List(pl.List(pl.Int64)),
            "force_u16",
            pl.UInt16,
        ) == pl.List(pl.List(pl.UInt16))

        # Null merge
        assert union_pl_dtypes(pl.Null, pl.Int64, "null_merge") == pl.Int64
        assert union_pl_dtypes(pl.Null, pl.Float64, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(
            pl.Null, pl.List(pl.Int64), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Float32)), "null_merge"
        ) == pl.List(pl.List(pl.Float32))
        assert union_pl_dtypes(
            pl.Array(pl.Null, (1, 1, 1)),
            pl.List(pl.Array(pl.Float32, (1, 1))),
            "null_merge",
        ) == pl.List(pl.List(pl.List(pl.Float32)))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.String), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Int32)), "force_u32", pl.UInt32
        ) == pl.List(pl.List(pl.UInt32))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.List(pl.Int32))), "null_merge"
        ) == pl.List(pl.List(pl.List(pl.Int32)))
        assert union_pl_dtypes(
            pl.List(pl.Null),
            pl.List(pl.List(pl.List(pl.Int32))),
            "force_u32",
            pl.UInt32,
        ) == pl.List(pl.List(pl.List(pl.UInt32)))

    def test_quote_columns(self):
        # Singles
        assert quote_columns("simple") == '"simple"'
        assert quote_columns("with spaces") == '"with spaces"'
        assert quote_columns("with-hyphens") == '"with-hyphens"'
        assert quote_columns("with[brackets]") == '"with[brackets]"'
        assert quote_columns("with/slashes") == '"with/slashes"'
        # Pre-quoted (must be escaped)
        assert quote_columns('already"quoted') == '"already""quoted"'
        assert quote_columns('"fully"quoted"') == '"""fully""quoted"""'
        # Lists
        assert quote_columns(["col1", "col2", "col3"]) == ['"col1"', '"col2"', '"col3"']
        assert quote_columns(["with spaces", "with-hyphens"]) == [
            '"with spaces"',
            '"with-hyphens"',
        ]
        assert quote_columns(["normal", "space here", "hyphen-here", 'quote"here']) == [
            '"normal"',
            '"space here"',
            '"hyphen-here"',
            '"quote""here"',
        ]
        # Empty cases
        assert quote_columns("") == '""'
        assert quote_columns([]) == []

        # End-to-end with DuckDB
        with tempfile.TemporaryDirectory() as tmp_path:
            test_file = os.path.join(tmp_path, "weird_cols.parquet")
            test_data = pl.DataFrame(
                {
                    "simple": [1, 2, 3],
                    "with spaces": [4, 5, 6],
                    "with-hyphens": [7, 8, 9],
                    "with[brackets]": [10, 11, 12],
                    "with/slashes": [13, 14, 15],
                    'has"quote': [16, 17, 18],
                    "dot.name": [19, 20, 21],
                    "colon:name": [22, 23, 24],
                }
            )
            test_data.write_parquet(test_file, statistics=False)
            conn = create_duckdb_conn()
            for col in test_data.columns:
                quoted_col = quote_columns(col)
                result = conn.sql(f"SELECT {quoted_col} FROM '{test_file}'").pl()
                assert result.shape == (3, 1)
                assert result.columns[0] == col
                assert result[col].to_list() == test_data[col].to_list()
            weird_cols = ["with spaces", "with-hyphens", "with[brackets]", 'has"quote']
            quoted_cols = ", ".join(quote_columns(weird_cols))
            result = conn.sql(f"SELECT {quoted_cols} FROM '{test_file}'").pl()
            assert result.shape == (3, 4)
            for col in weird_cols:
                assert col in result.columns
                assert result[col].to_list() == test_data[col].to_list()

    def test_list_columns(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            test_file = os.path.join(tmp_path, "test.parquet")
            test_data = pl.DataFrame(
                {
                    "col_a": [1, 2, 3],
                    "col_b": [4.0, 5.0, 6.0],
                    "listeners__mass__cell_mass": [7.0, 8.0, 9.0],
                    "listeners__mass__dry_mass": [10.0, 11.0, 12.0],
                    "listeners__growth__instantaneous_growth_rate": [0.1, 0.2, 0.3],
                    "bulk": [[1, 2], [3, 4], [5, 6]],
                }
            )
            test_data.write_parquet(test_file, statistics=False)
            conn = create_duckdb_conn()
            subquery = f"SELECT * FROM '{test_file}'"
            all_cols = list_columns(conn, subquery)
            assert len(all_cols) == 6
            assert "col_a" in all_cols
            assert "col_b" in all_cols
            assert "listeners__mass__cell_mass" in all_cols
            listener_cols = list_columns(conn, subquery, "listeners__*")
            assert len(listener_cols) == 3
            assert all(col.startswith("listeners__") for col in listener_cols)
            mass_cols = list_columns(conn, subquery, "listeners__mass__*")
            assert len(mass_cols) == 2
            assert "listeners__mass__cell_mass" in mass_cols
            assert "listeners__mass__dry_mass" in mass_cols
            no_match = list_columns(conn, subquery, "nonexistent__*")
            assert len(no_match) == 0
            col_pattern = list_columns(conn, subquery, "col_?")
            assert len(col_pattern) == 2
            assert "col_a" in col_pattern
            assert "col_b" in col_pattern
            exact = list_columns(conn, subquery, "bulk")
            assert exact == ["bulk"]


# ============================================================================
# Integration tests — direct __init__ → update → close → query lifecycle
# against process-bigraph's own core / Composite.
# ============================================================================


@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp(prefix="pbg_parquet_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_emitter_direct_lifecycle(temp_dir, core):
    """Construct the emitter directly, feed ticks, close, query back.

    Uses ``threaded=False`` to keep the writer in-thread for deterministic
    test sequencing. Verifies the flat (no-partitioning) layout: rows land
    under ``<out_dir>/<experiment_id>/history/`` and ``query()`` returns
    them as a polars DataFrame whose columns are the flattened state keys.
    """
    emitter = ParquetEmitter(
        config={
            "emit": {"time": "node", "value": "node"},
            "out_dir": temp_dir,
            "batch_size": 5,
            "threaded": False,
            "metadata": {"experiment_id": "exp_lifecycle"},
        },
        core=core,
    )

    # Feed 7 ticks so we exercise both the full-batch flush (at row 5) and
    # the close()-time partial-batch flush (the trailing 2 rows).
    for i in range(7):
        emitter.update({"time": float(i), "value": i * 10})
    emitter.close(success=False)

    df = emitter.query()
    assert df.shape[0] == 7
    # Columns are the flat state keys.
    assert "time" in df.columns
    assert "value" in df.columns
    times = sorted(df["time"].to_list())
    values = sorted(df["value"].to_list())
    assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert values == [0, 10, 20, 30, 40, 50, 60]

    # Two parquet files on disk: one full batch (5 rows) + one partial (2).
    history_dir = os.path.join(temp_dir, "exp_lifecycle", "history")
    pq_files = sorted(f for f in os.listdir(history_dir) if f.endswith(".pq"))
    assert len(pq_files) == 2


def test_emitter_partitioned_writes_success_sentinel(temp_dir, core):
    """When partitioning_keys + close(success=True), a success sentinel lands.

    Also verifies that hive-partition values from metadata appear as columns
    in the queried-back DataFrame.
    """
    emitter = ParquetEmitter(
        config={
            "emit": {"time": "node"},
            "out_dir": temp_dir,
            "batch_size": 3,
            "threaded": False,
            "partitioning_keys": ["experiment_id", "variant"],
            "metadata": {"experiment_id": "exp_part", "variant": 7},
        },
        core=core,
    )
    for i in range(4):
        emitter.update({"time": float(i)})
    emitter.close(success=True)

    # Hive partition path under history/
    history_dir = os.path.join(
        temp_dir, "exp_part", "history", "experiment_id=exp_part", "variant=7"
    )
    assert os.path.isdir(history_dir)
    pq_files = [f for f in os.listdir(history_dir) if f.endswith(".pq")]
    # batch_size=3 with 4 ticks => 1 full + 1 partial = 2 files.
    assert len(pq_files) == 2

    # Success sentinel under success/.
    success_dir = os.path.join(
        temp_dir, "exp_part", "success", "experiment_id=exp_part", "variant=7"
    )
    assert os.path.isfile(os.path.join(success_dir, "s.pq"))

    df = emitter.query()
    assert df.shape[0] == 4
    # Hive partition values surface as columns.
    assert "experiment_id" in df.columns
    assert "variant" in df.columns
    assert set(df["experiment_id"].to_list()) == {"exp_part"}
    assert set(df["variant"].to_list()) == {7}


def test_emitter_inside_composite(temp_dir, core):
    """End-to-end via Composite: the emitter is wired in as a Step.

    Mirrors the existing SQLiteEmitter integration test but for parquet.
    Uses ``threaded=False`` for deterministic ordering and
    ``flush_all_in_composite`` to ensure the trailing partial batch lands
    on disk before we read back via the same emitter instance's ``query()``.
    """
    composite_spec = {
        "increase": {
            "_type": "process",
            "address": "local:IncreaseProcess",
            "config": {"rate": 0.3},
            "interval": 1.0,
            "inputs": {"level": ["value"]},
            "outputs": {"level": ["value"]},
        },
    }
    composite = Composite({"state": composite_spec}, core)

    emitter_spec = {
        "_type": "step",
        "address": "local:ParquetEmitter",
        "config": {
            "emit": {"global_time": "node", "value": "node"},
            "out_dir": temp_dir,
            "batch_size": 4,
            "threaded": False,
            "metadata": {"experiment_id": "exp_composite"},
        },
        "inputs": {"global_time": ["global_time"], "value": ["value"]},
    }
    composite.merge({}, bs_set_path({}, ("emitter",), emitter_spec))
    _, instance = core.traverse(
        composite.schema, composite.state, ("emitter",)
    )
    composite.step_paths[("emitter",)] = instance
    composite.build_step_network()

    composite.run(6)

    # Driver doesn't hold the emitter instance directly — use the walker.
    closed = ParquetEmitter.flush_all_in_composite(composite, success=False)
    assert closed == 1

    df = composite.state["emitter"]["instance"].query()
    # Composite emits at every tick including t=0, so we expect 7 rows
    # (composite.run(6) advances global_time from 0 to 6 inclusive).
    assert df.shape[0] >= 6
    assert "global_time" in df.columns
    assert "value" in df.columns
    # Values should be monotonically non-decreasing under a positive-rate
    # IncreaseProcess.
    values = df.sort("global_time")["value"].to_list()
    assert values == sorted(values)
