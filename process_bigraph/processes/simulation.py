"""A whole simulation as one node of an outer step network.

A simulation is itself a step network, so a study is a **higher-order DAG**:
the simulation is a single node, and the flush entities — visualizations,
analyses, report cards — are ordinary steps downstream of it.

That framing is what makes a flush step fire *once*. Its ``results`` input is
unsatisfied until the simulation node's update returns, and the simulation's
per-tick stepping happens *inside* the node rather than beside it, so the
flush steps are never siblings of the simulation's own steps. No completion
phase, no marker, no scheduler special case — ordinary producer/consumer
ordering does all of it.

Compare ``parameter_scan.RunProcess``, which is the same shape specialized to
one process address and materialized observable timeseries.
``SimulationStep`` takes a whole composite document and emits the emitter's
:class:`~process_bigraph.emitter.EmitterResults` **handle** — a reference, so
the outer network stays cheap however large the run was.
"""

import copy

from process_bigraph.composite import Step, Composite


class SimulationStep(Step):
    """Run an inner composite to completion and emit its results handle.

    Config:
      ``state``    — the inner composite's state document (processes, stores,
                     and at least one emitter whose ``results`` is wired).
      ``runtime``  — how long to run the inner simulation.
      ``emitter``  — optional ``/``-joined path naming which emitter supplies
                     ``results``; required only when the inner composite has
                     more than one.
      ``document`` — optional extra composite document keys (``bridge``,
                     ``global_time_precision``, …) merged with ``state``.

    The built composite is kept on ``self.composite`` after the first update,
    so a caller can inspect the finished simulation.
    """

    config_schema = {
        'state': 'quote',
        'runtime': 'float',
        'emitter': 'maybe[string]',
        'document': 'quote'}

    def initialize(self, config):
        self.composite = None

    def inputs(self):
        return {}

    def outputs(self):
        return {'results': 'node'}

    def _select(self, handles):
        """Pick the emitter whose handle becomes ``results``."""
        if not handles:
            raise ValueError(
                'SimulationStep: the inner composite produced no results — '
                'it needs an emitter with its `results` port wired, e.g. '
                "`'outputs': {'results': ['results']}`.")

        declared = self.config.get('emitter')
        if declared:
            wanted = tuple(part for part in str(declared).split('/') if part)
            if wanted not in handles:
                available = sorted('/'.join(path) for path in handles)
                raise ValueError(
                    f'SimulationStep: no emitter at {declared!r} — '
                    f'the inner composite has {available}.')
            return handles[wanted]

        if len(handles) > 1:
            available = sorted('/'.join(path) for path in handles)
            raise ValueError(
                f'SimulationStep: the inner composite has {len(handles)} '
                f'emitters ({available}); name the one that supplies '
                f'`results` with the `emitter` config.')

        return next(iter(handles.values()))

    def update(self, state):
        document = dict(self.config.get('document') or {})
        document['state'] = copy.deepcopy(self.config['state'])

        self.composite = Composite(document, core=self.core)
        self.composite.run(self.config['runtime'])

        # `finalize` is where a composite produces its completion-time
        # outputs — the emitters' durable handles.
        return {'results': self._select(self.composite.finalize())}
