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

from bigraph_schema.assembly import fill_sites, interfaces


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
    """
    filled = fill_template(core, template, bindings)

    unfilled = required_open_sites(filled)
    if unfilled:
        named = ', '.join(repr('/'.join(path)) for path in sorted(unfilled))
        raise ValueError(
            f'template is not ground — required site(s) left unfilled: '
            f'{named}')

    return core.render(filled)


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
    return core.render(pruned), blocked
