"""Discover composite specs across installed bigraph-schema-dependent packages.

A composite spec is any file matching *.composite.yaml or *.composite.json inside
an installed package whose distribution declares bigraph-schema as a requirement.
"""
from __future__ import annotations
import importlib
import importlib.metadata
import sys
from pathlib import Path

from process_bigraph.composite_spec import CompositeSpec


_GLOB_PATTERNS = ("*.composite.yaml", "*.composite.yml", "*.composite.json")


def _is_bigraph_schema_lib(dist) -> bool:
    if dist.metadata["Name"] == "bigraph-schema":
        return True
    reqs = dist.requires or []
    return any("bigraph-schema" in r for r in reqs)


def discover_composites(extra_search_paths: list[Path] | None = None) -> dict[str, dict]:
    """Walk installed bigraph-schema-dependent packages + optional extra dirs.

    Returns {composite_id: spec_dict}, where composite_id is a stable identifier
    of the form '<package>.<file-stem>' or '<package>.<sub.path>.<file-stem>'.

    extra_search_paths: additional directories to scan (e.g. workspace-local
    composites/ dirs).
    """
    specs: dict[str, dict] = {}

    # 1. Installed packages.
    for dist in importlib.metadata.distributions():
        if not _is_bigraph_schema_lib(dist):
            continue
        # Find every importable package this distribution declares.
        top_level = dist.read_text("top_level.txt") or ""
        for pkg_name in top_level.strip().splitlines():
            if not pkg_name:
                continue
            try:
                mod = importlib.import_module(pkg_name)
            except ImportError:
                continue
            if not getattr(mod, "__file__", None):
                continue
            pkg_root = Path(mod.__file__).parent
            for pattern in _GLOB_PATTERNS:
                for path in pkg_root.rglob(pattern):
                    spec_id = _make_spec_id(pkg_name, pkg_root, path)
                    try:
                        # Delegate parsing + validation to the unified engine;
                        # to_dict() serialises to a plain dict for downstream.
                        specs[spec_id] = CompositeSpec.from_file(path).to_dict()
                    except Exception as e:
                        # Skip malformed specs; log to stderr
                        print(f"warning: skipping {path}: {e}", file=sys.stderr)

    # 2. Extra paths (e.g. workspace-local).
    for extra in (extra_search_paths or []):
        if not extra.is_dir():
            continue
        for pattern in _GLOB_PATTERNS:
            for path in extra.rglob(pattern):
                spec_id = _make_spec_id("local", extra, path)
                try:
                    specs[spec_id] = CompositeSpec.from_file(path).to_dict()
                except Exception as e:
                    print(f"warning: skipping {path}: {e}", file=sys.stderr)

    return specs


def discover_all(
    extra_search_paths: list[Path] | None = None,
    extra_packages: list[str] | None = None,
) -> dict[str, dict]:
    """Return both static specs and composite generators, tagged with ``kind``.

    Spec entries get ``kind: spec`` and pass through the existing spec
    payload. Generator entries get ``kind: generator`` and a compact dict
    {name, description, parameters, module} so dashboards can render them
    without holding a reference to the live function object.

    Discovery imports the host packages of every generator (see
    :func:`process_bigraph.composite_generator.discover_generators`). If
    you need the no-import safety property, call ``discover_composites``
    directly.
    """
    # Deferred: discover_generators triggers package imports as a
    # side-effect. Keeping it local ensures that cost is paid only when
    # discover_all is explicitly called.
    from process_bigraph.composite_generator import discover_generators

    specs = discover_composites(extra_search_paths=extra_search_paths)
    out: dict[str, dict] = {
        sid: {"kind": "spec", **s} for sid, s in specs.items()
    }
    for gid, entry in discover_generators(
            extra_packages=extra_packages).items():
        if gid in out:
            import warnings
            warnings.warn(
                f"discover_all: generator {gid!r} collides with a spec of "
                "the same id; generator wins. Rename one to avoid ambiguity.",
                stacklevel=2,
            )
        out[gid] = {
            "kind": "generator",
            "name": entry.name,
            "description": entry.description,
            "parameters": entry.parameters,
            "visualizations": list(entry.visualizations),
            "emitters": list(entry.emitters),
            "module": entry.module,
            "default_n_steps": entry.default_n_steps,
        }
    return out


def _make_spec_id(pkg_name: str, pkg_root: Path, file_path: Path) -> str:
    """Stable identifier for a composite spec: 'pkg.subdir.subdir.stem'."""
    rel = file_path.relative_to(pkg_root)
    # file_path stem strips both .yaml/.json AND the .composite part if .composite.yaml
    stem = file_path.name
    for suffix in (".composite.yaml", ".composite.yml", ".composite.json"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    parts = [pkg_name] + list(rel.parts[:-1]) + [stem]
    return ".".join(parts)
