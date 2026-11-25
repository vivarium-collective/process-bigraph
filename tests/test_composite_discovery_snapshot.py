"""``discover_composites`` walks the distributions as they were when it started.

``importlib.metadata.distributions()`` reads ``sys.meta_path`` live, and discovery imports each package it walks. A
package that installs a meta-path finder at import (the pbg -> viva rename shims do ``sys.meta_path.insert(0, ...)``)
shifts the list under the walk, so the walk visits the path finder again and yields every distribution again. In a
workbench env worker this never ended (the composite listing of the v2ecoli workspace spun at 100% CPU).
"""
import sys
import textwrap

from process_bigraph import composite_discovery
from process_bigraph.composite_spec import CompositeSpec


def _make_package(root, name):
    """A distribution that requires bigraph-schema, ships one composite spec, and inserts a meta-path finder at
    import, as the rename shims do."""
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent("""
        import importlib.abc
        import sys


        class _Redirect(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                return None


        sys.meta_path.insert(0, _Redirect())
    """))
    (pkg / "demo.composite.yaml").write_text("name: demo\nstate: {}\n")
    info = root / f"{name}-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\nRequires-Dist: bigraph-schema\n")
    (info / "top_level.txt").write_text(f"{name}\n")


def test_a_package_inserting_a_finder_at_import_is_walked_once(tmp_path, monkeypatch):
    _make_package(tmp_path, "redirecting_pkg")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))  # the package's finder goes away with the test
    monkeypatch.delitem(sys.modules, "redirecting_pkg", raising=False)

    parsed = []
    original = CompositeSpec.from_file.__func__

    def counting(cls, path, *args, **kwargs):
        if "redirecting_pkg" in str(path):
            parsed.append(path)
        return original(cls, path, *args, **kwargs)

    monkeypatch.setattr(CompositeSpec, "from_file", classmethod(counting))
    specs = composite_discovery.discover_composites()

    assert len(parsed) == 1, f"redirecting_pkg's spec was parsed {len(parsed)} times"
    assert any(key.startswith("redirecting_pkg") for key in specs)
