"""Study and investigation templates.

A **template** is a document that is not ground — it has open *sites*, the
place-graph holes bigraph-schema's ``fill_sites`` fills. A **study template**
fixes its analysis network and leaves the model as a site; an **investigation
template** has one site per member study.

Two things here that the bigraph-schema helpers do not cover:

- **``template_document``** builds a document process-bigraph can actually
  consume. ``assembly.build`` finishes with ``core.fill``, which *realizes*
  every edge — it instantiates processes and it cannot walk a process's
  ``config`` (a raw value where ``default`` expects a schema). ``Composite``
  does its own realization anyway, so the native path is
  ``fill_sites`` → ``render`` → ``Composite``.

- **``prune_open_regions``** makes "a failed prerequisite is not built" express
  a *region*. A site left open is not a local condition: ``Composite`` rejects
  a document with *any* open required site, so leaving one open makes the whole
  investigation unbuildable rather than skipping one member. Pruning drops the
  regions that are still open and reports them as blocked, leaving the rest
  ground and runnable.
"""

import copy
import json
from pathlib import Path

from bigraph_schema.assembly import fill_sites, interfaces


# ── on-disk templates ───────────────────────────────────────────────
#
# A template is authorable as a file: a ``*.template.{yaml,json}`` document
# with sites written in place. Loading is *just* ``core.access`` on the parsed
# document — access reconstitutes each site as a place-graph hole, and
# bigraph-schema 1.4.4's ``_sort`` round-trip means a site's value type / face /
# default survives the file. The loaded document is the same thing the
# Python-built templates in the tests are: it hands straight to
# ``fill_template`` / ``template_document`` / ``investigation_document``.
#
# A study template and an investigation template are the *same* on-disk shape —
# a document with sites — so one loader reads both; the caller picks
# ``template_document`` (study) or ``investigation_document`` (investigation).

TEMPLATE_SUFFIXES = ('.template.yaml', '.template.yml', '.template.json')
"""Recognised on-disk template extensions (spec §6 authoring format)."""


def _read_mapping(path):
    """Parse a YAML or JSON document to a dict, chosen by extension."""
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        raw = json.loads(text)
    else:
        import yaml
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        kind = type(raw).__name__
        raise ValueError(
            f'template {path} must parse to a mapping, got {kind}')
    return raw


def load_template(path, core):
    """Read a ``*.template.{yaml,json}`` document from disk into a template.

    Parses the file (YAML or JSON, by extension) and runs the result through
    ``core.access``, which reconstitutes every site as an open place-graph
    hole. Because ``access`` round-trips ``_sort`` (the Layer-1 fix in
    bigraph-schema 1.4.4), a value site's type, a face site's interface and a
    site's ``_default`` all survive the file intact — the loaded document has
    exactly the ``open_sites`` the file declared.

    The returned document is a template like any other: hand it to
    ``fill_template`` / ``template_document`` (a study) or
    ``investigation_document`` (an investigation, one member per site). Both
    file kinds load through this one function; only the caller's follow-on call
    differs. Any process/step edges named in the file must already be
    registered on ``core``.
    """
    return core.access(_read_mapping(path))


def save_template(template, path, core):
    """Write a template document to disk so it re-loads with identical sites.

    Rendered with ``defaults=True`` on purpose. A plain ``render`` drops each
    site's ``_default`` and stringizes a face ``_sort`` — after round-tripping
    through the file an optional site would come back **required** and a face
    would be flattened, so ``load_template`` would read back a *different*
    template. ``defaults=True`` is what makes the on-disk round-trip identity
    (the same subtlety ``template_document`` documents for the fill path).

    Returns the ``Path`` written.
    """
    path = Path(path)
    rendered = core.render(template, defaults=True)
    if path.suffix.lower() == '.json':
        text = json.dumps(rendered, indent=2, sort_keys=True)
    else:
        import yaml
        text = yaml.safe_dump(rendered, sort_keys=True)
    path.write_text(text, encoding='utf-8')
    return path


def open_sites(document):
    """The paths of every site still open in ``document``."""
    return [tuple(path) for path, _site in interfaces(document)[0]._places]


def required_open_sites(document):
    """Open sites that have no ``_default`` — the ones that block a build."""
    return [
        tuple(path)
        for path, site in interfaces(document)[0]._places
        if getattr(site, '_default', None) is None]


def is_ground_document(document):
    """True when nothing is left to fill — the runnable predicate."""
    return not required_open_sites(document)


def fill_template(core, template, bindings=None):
    """Fill a template's sites. Sites left unbound stay open, so the result
    may still be a template — filling is incremental."""
    return fill_sites(core, template, bindings or {})


def template_document(core, template, bindings=None):
    """Fill a template and render the document ``Composite`` consumes.

    Raises when a required site is left unfilled, naming it — the same
    condition ``Composite`` enforces, reported before construction so the
    caller sees which hole is empty rather than a constructor error.

    Rendered with ``defaults=True``: filling a *value* site puts the value on
    its sort's ``_default``, and a plain render drops defaults — a filled
    ``float`` site would come back as the bare string ``'float'`` and the
    value would be silently lost.
    """
    filled = fill_template(core, template, bindings)

    unfilled = required_open_sites(filled)
    if unfilled:
        named = ', '.join(repr('/'.join(path)) for path in sorted(unfilled))
        raise ValueError(
            f'template is not ground — required site(s) left unfilled: '
            f'{named}')

    return core.render(filled, defaults=True)


def prune_open_regions(document, member_depth=1):
    """Drop every region that still contains an open site.

    Returns ``(pruned_document, blocked)`` where ``blocked`` lists the pruned
    region paths. ``member_depth`` says how deep a *member* sits: for an
    investigation shaped ``{investigation: {study_A: …, study_B: …}}`` the
    members are at depth 2.

    This is what lets a gate's verdict decide membership. A member whose site
    was never filled is not merely unfilled — it is **absent** from the built
    document, so it is never constructed and never runs, while its siblings
    still do.
    """
    pruned = copy.deepcopy(document)

    blocked = sorted({
        path[:member_depth] for path in required_open_sites(pruned)})

    for region in blocked:
        parent = pruned
        for step in region[:-1]:
            parent = parent[step]
        if isinstance(parent, dict):
            parent.pop(region[-1], None)

    return pruned, ['/'.join(region) for region in blocked]


def investigation_document(core, template, bindings=None, member_depth=1):
    """Build an investigation, skipping members whose sites are still open.

    Returns ``(document, blocked)``. Filling a member's site admits it to the
    run; leaving it open keeps it out. That is gating expressed as filling
    rather than as scheduling: the engine never has to decide *not* to run
    something, because what was not filled is not in the document.
    """
    filled = fill_template(core, template, bindings)
    pruned, blocked = prune_open_regions(filled, member_depth=member_depth)
    return core.render(pruned, defaults=True), blocked


# ── partial graph triggering: pull-or-compute ───────────────────────
#
# A cached study result is simply a **filled** results site. Nothing new is
# scheduled: an open results site means the study computes, a filled one means
# it does not and its consumers read the cached handle instead. `trigger` is
# the operation that decides which, using the same fill/prune law gating uses.

STUDY_META = '_study'
"""Per-member metadata: ``{id, config, inputs: [{artifact, from}], kind}``."""


def study_members(document):
    """Every member of an investigation document, by name.

    A member is a top-level region carrying ``_study`` metadata.
    """
    return {
        name: region for name, region in document.items()
        if isinstance(region, dict) and isinstance(region.get(STUDY_META), dict)}


def _meta(region, key, default=None):
    return (region.get(STUDY_META) or {}).get(key, default)


def study_ancestors(document, target):
    """``target``'s transitive prerequisites, nearest last.

    Edges come from each member's ``inputs: [{artifact, from}]`` — the same
    declaration the investigation format already carries, so the DAG is read
    rather than restated.
    """
    members = study_members(document)
    if target not in members:
        raise KeyError(
            f'no study {target!r} in the document — members are '
            f'{sorted(members)}')

    order, seen = [], set()

    def walk(name):
        if name in seen:
            return
        seen.add(name)
        for edge in _meta(members.get(name, {}), 'inputs', []) or []:
            upstream = edge.get('from')
            if upstream and upstream in members:
                walk(upstream)
                if upstream not in order:
                    order.append(upstream)

    walk(target)
    return order


def study_address(document, name, *, commit=''):
    """The content address of a member's artifact.

    A function of the member's id, its config, its inputs' addresses and the
    workspace commit — so it is worktree-independent and an upstream change
    invalidates everything downstream of it.
    """
    from process_bigraph.artifacts import artifact_id

    members = study_members(document)
    region = members[name]

    input_ids = [
        study_address(document, edge['from'], commit=commit)
        for edge in (_meta(region, 'inputs', []) or [])
        if edge.get('from') in members]

    return artifact_id(
        composite_id=_meta(region, 'id', name),
        config=_meta(region, 'config', {}) or {},
        input_ids=input_ids,
        commit=commit)


def cached_node(ref):
    """The step that replaces a member's simulation when it is pulled."""
    return {
        '_type': 'step',
        'address': 'local:CachedResults',
        'config': {'artifact_ref': ref},
        'inputs': {},
        'outputs': {'results': ['results']}}


def trigger(document, target, *, on_missing='error', commit='',
            root=None, sim_key='sim'):
    """Build the sub-document that runs ``target``.

    Prerequisites already in the artifact store are **pulled**: their
    simulation node is swapped for a ``CachedResults`` bound to the resolved
    reference, so they are ground and do not run. The target is left open, so
    it computes. Everything that is neither is **pruned** — absent from the
    document, never constructed.

    "Run one study" and "rerun this study reusing its upstream" are the same
    call with different targets.

    ``on_missing='error'`` (default) refuses to silently recompute a
    prerequisite the caller meant to reuse, naming it instead;
    ``on_missing='compute'`` leaves it open so it computes too.

    Returns ``(document, report)`` where report is
    ``{'target', 'pulled', 'computed', 'pruned'}``.
    """
    from process_bigraph.artifacts import (
        ARTIFACT_ROOT, ArtifactRef, artifact_exists, artifact_store)

    if on_missing not in ('error', 'compute'):
        raise ValueError(
            f"on_missing must be 'error' or 'compute', got {on_missing!r}")
    root = ARTIFACT_ROOT if root is None else root

    members = study_members(document)
    ancestors = study_ancestors(document, target)
    keep = set(ancestors) | {target}

    built = copy.deepcopy(document)
    pulled, computed, missing = [], [], []

    for name in ancestors:
        address = study_address(document, name, commit=commit)
        if artifact_exists(address, root):
            ref = ArtifactRef(
                kind=_meta(members[name], 'kind', 'trajectory'),
                hash=address,
                store=artifact_store(address, root),
                context={'study': name})
            built[name][sim_key] = cached_node(ref.to_dict())
            pulled.append(name)
        elif on_missing == 'compute':
            computed.append(name)
        else:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f'cannot trigger {target!r}: prerequisite(s) '
            f'{", ".join(repr(n) for n in missing)} have no cached artifact. '
            f'Run them first, or pass on_missing="compute" to compute them '
            f'as part of this trigger.')

    pruned = sorted(set(members) - keep)
    for name in pruned:
        built.pop(name, None)

    return built, {
        'target': target,
        'pulled': pulled,
        'computed': computed + [target],
        'pruned': pruned}
