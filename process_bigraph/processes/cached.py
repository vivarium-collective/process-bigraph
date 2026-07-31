"""The pull half of pull-or-compute.

``SimulationStep`` computes a study's results by running it. ``CachedResults``
produces the same thing by resolving an artifact that already exists. They
share the ``{results: node}`` face and both produce at finalize, so nothing
downstream can tell which one it is wired to — a study that was pulled and a
study that just ran look identical from the outside.

That indistinguishability is the contract, not an implementation detail: it is
what lets ``trigger`` swap one for the other in a document without rewiring
anything.
"""

from __future__ import annotations

from process_bigraph.composite import Step
from process_bigraph.artifacts import ArtifactResults


class CachedResults(Step):
    """Resolve a content-addressed artifact and emit its handle.

    Never runs a simulation. Config is the ``artifact_ref``
    (``{kind, hash, store, context}``) that ``trigger`` bound when it found
    the artifact in the store.
    """

    config_schema = {'artifact_ref': 'quote'}

    def inputs(self):
        return {}

    def outputs(self):
        return {'results': 'node'}

    def results(self, path=None, context=None) -> ArtifactResults:
        """The handle for the stored artifact.

        Named to match the emitter contract — ``Composite.finalize()`` asks
        every edge for ``results()``, so a pulled study surfaces its handle by
        exactly the route a computed one does.
        """
        ref = self.config.get('artifact_ref')
        if not ref:
            raise ValueError(
                'CachedResults has no `artifact_ref` — it can only pull an '
                'artifact that trigger() already resolved.')
        return ArtifactResults(ref, context=context)

    def update(self, state) -> dict:
        """Emit the handle the way ``SimulationStep`` does.

        This is where indistinguishability is actually won. An *emitter*
        produces ``results`` at finalize, because its results only exist once
        the run is over — but a study node is a producer *in the step DAG*,
        and ``SimulationStep.update()`` returns the handle so its consumers
        are ordered after it. A ``CachedResults`` that produced only at
        finalize would leave those consumers unordered: they would fire first
        and read an empty store. Matching ``SimulationStep`` — not matching
        the emitter — is what the contract requires.
        """
        return {'results': self.results()}

    def finalize(self, path=None, context=None) -> dict:
        """Also answer the completion-time route, so a composite that asks
        every edge for its results at finalize gets this one too."""
        return {'results': self.results(path=path, context=context)}
