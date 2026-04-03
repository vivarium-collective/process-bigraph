"""Dynamic structure process for testing structural changes during simulation.

DynamicWorker is a Process that performs structural operations (spawn, remove,
rewire, value changes) based on deterministic conditions from its input state.
Used to test cache invalidation when wiring changes mid-simulation.
"""

from process_bigraph.composite import Process


class DynamicWorker(Process):
    """Process that modifies pool structure based on input conditions.

    Reads from 'sources' (the entire pool map) and 'self_value'.
    Writes structural changes (_add, _remove) to 'targets' (the pool map)
    and value deltas to 'self_value'.

    Operations in priority order:
    1. Self-remove when projected value drops below threshold_remove
    2. Remove sources whose values are below threshold_remove
    3. Rewire: replace self in pool with new config (tests cache invalidation)
    4. Spawn: add a new agent to the pool
    5. Grow: increment self_value by growth_rate * interval
    """

    config_schema = {
        'process_id': 'string{}',
        'growth_rate': 'float{1.0}',
        'spawn_growth_rate': 'float{0.8}',
        'propensity_spawn': 'float{1.0}',
        'propensity_remove': 'float{1.0}',
        'propensity_rewire': 'float{0.0}',
        'threshold_spawn': 'float{3.0}',
        'threshold_remove': 'float{-3.0}',
        'threshold_rewire': 'float{4.0}',
        'max_pool_size': 'integer{15}',
        'spawn_value': 'float{0.5}',
    }

    _counter = 0

    def inputs(self):
        return {
            'sources': 'map[value:float]',
            'self_value': 'float'}

    def outputs(self):
        return {
            'targets': 'map[value:float]',
            'self_value': 'float'}

    def _source_values(self, sources):
        """Extract {agent_id: value} from pool state, excluding self.

        Handles both dict sources (entire pool) and scalar sources
        (single peer value after rewiring).
        """
        my_id = self.config['process_id']
        if isinstance(sources, (int, float)):
            return {'_peer': float(sources)}
        if not isinstance(sources, dict):
            return {}
        result = {}
        for k, v in sources.items():
            if k == my_id:
                continue
            if isinstance(v, dict) and 'value' in v:
                result[k] = float(v['value'])
            elif isinstance(v, (int, float)):
                result[k] = float(v)
        return result

    def _make_spawn_config(self, new_id):
        new_growth = self.config['spawn_growth_rate']
        next_spawn_growth = new_growth * 0.625
        if 0 < next_spawn_growth < 0.6:
            next_spawn_growth = -0.5

        return {
            'process_id': new_id,
            'growth_rate': new_growth,
            'spawn_growth_rate': next_spawn_growth,
            'propensity_spawn': 1.0 if new_growth > 0 else 0.0,
            'propensity_remove': 1.0,
            'propensity_rewire': 0.0,
            'threshold_spawn': self.config['threshold_spawn'],
            'threshold_remove': self.config['threshold_remove'],
            'threshold_rewire': self.config['threshold_rewire'],
            'max_pool_size': self.config['max_pool_size'],
            'spawn_value': self.config['spawn_value'],
        }

    def _make_agent(self, agent_id, value, config):
        return {
            'value': value,
            'worker': {
                'address': 'local:DynamicWorker',
                'config': config,
                'inputs': {
                    'sources': ['..'],
                    'self_value': ['value']},
                'outputs': {
                    'targets': ['..'],
                    'self_value': ['value']}}}

    def update(self, state, interval):
        self_val = state.get('self_value', 0.0)
        sources = state.get('sources', {})
        source_vals = self._source_values(sources)
        source_sum = sum(source_vals.values()) if source_vals else 0.0
        source_count = len(source_vals)

        delta = self.config['growth_rate'] * interval
        projected = self_val + delta
        my_id = self.config['process_id']

        # --- Priority 1: Self-remove when value too negative ---
        if (self.config['propensity_remove'] > 0
                and projected * self.config['propensity_remove']
                < self.config['threshold_remove']):
            return {
                'self_value': delta,
                'targets': {'_remove': [my_id]}}

        # --- Priority 2: Remove sources with very negative values ---
        removals = [
            sid for sid, sv in source_vals.items()
            if (self.config['propensity_remove'] > 0
                and sv * self.config['propensity_remove']
                < self.config['threshold_remove'])]
        if removals:
            return {
                'self_value': delta,
                'targets': {'_remove': removals}}

        # --- Priority 3: Rewire (replace self with genuinely different wires) ---
        # After rewiring, self_value output writes to a peer's value instead
        # of own value. This genuinely changes compiled wire paths and tests
        # that the view/project cache is correctly invalidated.
        if (self.config['propensity_rewire'] > 0
                and source_count > 0
                and source_sum * self.config['propensity_rewire']
                > self.config['threshold_rewire']):
            best_peer = max(source_vals, key=source_vals.get)
            config = {
                'process_id': my_id,
                'growth_rate': self.config['growth_rate'],
                'spawn_growth_rate': self.config['spawn_growth_rate'],
                'propensity_spawn': self.config['propensity_spawn'],
                'propensity_remove': self.config['propensity_remove'],
                'propensity_rewire': 0.0,  # prevent immediate re-rewire
                'threshold_spawn': self.config['threshold_spawn'],
                'threshold_remove': self.config['threshold_remove'],
                'threshold_rewire': self.config['threshold_rewire'],
                'max_pool_size': self.config['max_pool_size'],
                'spawn_value': self.config['spawn_value'],
            }
            rewired = {
                'value': projected,
                'worker': {
                    'address': 'local:DynamicWorker',
                    'config': config,
                    'inputs': {
                        'sources': ['..'],
                        'self_value': ['value']},
                    'outputs': {
                        'targets': ['..'],
                        'self_value': ['..', best_peer, 'value']}}}
            return {
                'self_value': 0.0,
                'targets': {'_add': [(my_id, rewired)]}}

        # --- Priority 4: Spawn new agent ---
        if (self.config['propensity_spawn'] > 0
                and projected * self.config['propensity_spawn']
                > self.config['threshold_spawn']
                and source_count + 1
                < self.config['max_pool_size']):
            DynamicWorker._counter += 1
            new_id = f"{my_id}_{DynamicWorker._counter}"

            spawn_config = self._make_spawn_config(new_id)
            new_agent = self._make_agent(
                new_id,
                self.config['spawn_value'],
                spawn_config)

            return {
                'self_value': self.config['spawn_value'] - self_val,
                'targets': {'_add': [(new_id, new_agent)]}}

        # --- Default: grow value ---
        return {'self_value': delta, 'targets': {}}
