"""
Reaction Step
=============

A Step that fires Milner-style reaction rules on its input state.

Rules are ``ReactionRule`` objects from ``bigraph_schema.assembly``.
The step searches the input subtree for redex matches, fires one
(deterministically or stochastically), and returns the modified
subtree as its output.

Usage::

    from bigraph_schema.schema import Site
    from bigraph_schema.assembly import ReactionRule

    b3 = ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        label='B3: agent enters room')

    composite_spec = {
        'state': {
            'building': { ... }},
        'reaction': {
            '_type': 'step',
            'address': 'local:!process_bigraph.processes.reaction.ReactionStep',
            'config': {
                'rules': [b3],
                'mode': 'deterministic'},
            'inputs': {'state': ['building']},
            'outputs': {'state': ['building']}}}
"""

import random

from process_bigraph.composite import Step

from bigraph_schema.assembly import (
    ReactionRule,
    find_matches,
    fire_rule,
    run_reactions,
    ACTIVE, PASSIVE, ATOMIC,
)


class ReactionStep(Step):
    """A Step that applies reaction rules to a state subtree.

    Config:
        rules: List of ``ReactionRule`` objects.
        control_status: Dict mapping control names to
            ``'active'``/``'passive'``/``'atomic'``. Default: all active.
        mode: ``'deterministic'`` (first match wins) or
            ``'stochastic'`` (Gillespie-weighted by ``rule.rate``).
        seed: Optional RNG seed for stochastic mode.
    """

    config_schema = {}

    def initialize(self, config):
        self.rules = config.get('rules', [])
        self.control_status = config.get('control_status', {})
        self.mode = config.get('mode', 'deterministic')
        seed = config.get('seed', None)
        self.rng = random.Random(seed)

    def inputs(self):
        return {'state': 'tree[node]'}

    def outputs(self):
        return {'state': 'tree[node]'}

    def update(self, state):
        subtree = state.get('state', {})

        if self.mode == 'stochastic':
            # Collect all candidates
            candidates = []
            for rule in self.rules:
                matches = find_matches(
                    subtree, rule.redex, self.control_status)
                rate = rule.rate if rule.rate is not None else 1.0
                for match in matches:
                    candidates.append((rule, match, rate))

            if not candidates:
                return {}

            total = sum(r for _, _, r in candidates)
            pick = self.rng.random() * total
            cumulative = 0.0
            for rule, match, rate in candidates:
                cumulative += rate
                if cumulative >= pick:
                    new_state, _ = fire_rule(
                        subtree, rule, self.control_status)
                    if new_state is not subtree:
                        return {'state': new_state}
                    break
            return {}

        else:
            # Deterministic: first rule, first match
            for rule in self.rules:
                new_state, match = fire_rule(
                    subtree, rule, self.control_status)
                if match is not None:
                    return {'state': new_state}
            return {}
