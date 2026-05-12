"""
Bigraphical Reactive System
===========================

A Process that fires Milner-style reaction rules on each tick.

Whereas ``ReactionStep`` is a Step (fires only when its inputs
change, until quiescence), ``BigraphicalReactiveSystem`` is a
Process that fires on a regular time interval — suitable for
time-stepped traces where rules have no fixed point (e.g.
molecules continually wandering between compartments), and for
proper Gillespie SSA τ-leap stepping where the wall-clock
interval bounds a variable number of firings per tick.
"""
import math
import random

from process_bigraph.composite import Process

from bigraph_schema.assembly import (
    ReactionRule, find_matches, fire_rule)


class BigraphicalReactiveSystem(Process):
    """A Process that fires Milner-style reaction rules on each tick.

    Config:
        rules: List of ``ReactionRule`` objects. Each rule's
            ``rate`` field is treated as a *propensity coefficient*
            in the chemical-master-equation sense: the propensity
            of rule ``R`` is ``rule.rate * |matches(R)|``, so the
            firing distribution maps to mass-action over the match
            multiset.
        mode: One of:

            - ``'deterministic'`` — first matching rule wins; one
              firing per tick.
            - ``'stochastic'`` — one firing per tick, picked
              Gillespie-style by per-match propensity.
            - ``'gillespie'`` — proper SSA τ-leap: sample
              exponential waits with parameter ``λ = Σ k_i·|m_i|``
              until ``t ≥ interval``. Fires zero or more rules
              per tick depending on the propensity.

        seed: RNG seed (stochastic / gillespie modes).
        max_per_tick: Cap on firings per tick (default 1 for the
            non-Gillespie modes; Gillespie defaults to no cap).
    """

    config_schema = {}

    def initialize(self, config):
        self.rules = config.get('rules', [])
        self.mode = config.get('mode', 'deterministic')
        self.rng = random.Random(config.get('seed', 0))
        default_cap = (
            10**9 if self.mode == 'gillespie' else 1)
        self.max_per_tick = int(config.get('max_per_tick', default_cap))
        self.fired_log = []  # (sim_time, rule_label, match_path)

    def inputs(self):
        return {'state': 'tree[node]'}

    def outputs(self):
        return {'state': 'overwrite[tree[node]]'}

    def update(self, state, interval):
        subtree = state.get('state', {})
        if self.mode == 'gillespie':
            new_subtree, fired_any = self._gillespie_step(
                subtree, interval)
            return {'state': new_subtree} if fired_any else {}

        any_fired = False
        for _ in range(self.max_per_tick):
            new_subtree, label, path = self._fire_one(subtree)
            if label is None:
                break
            self.fired_log.append((interval, label, path))
            subtree = new_subtree
            any_fired = True
        if not any_fired:
            return {}
        return {'state': subtree}

    def _enumerate_candidates(self, subtree):
        candidates = []
        for rule in self.rules:
            matches = find_matches(subtree, rule.redex)
            rate = rule.rate if rule.rate is not None else 1.0
            for i, m in enumerate(matches):
                candidates.append((rule, i, m, rate))
        return candidates

    def _pick_candidate(self, candidates):
        if not candidates:
            return None
        total = sum(r for _, _, _, r in candidates)
        if total <= 0:
            return None
        pick = self.rng.random() * total
        cum = 0.0
        for rule, idx, match, rate in candidates:
            cum += rate
            if cum >= pick:
                return rule, idx, match
        rule, idx, match, _ = candidates[-1]
        return rule, idx, match

    def _fire_one(self, subtree):
        if self.mode == 'stochastic':
            candidates = self._enumerate_candidates(subtree)
            picked = self._pick_candidate(candidates)
            if picked is None:
                return subtree, None, None
            rule, idx, match = picked
            new_state, _ = fire_rule(subtree, rule, match_index=idx)
            return new_state, rule.label, match.path
        else:
            for rule in self.rules:
                new_state, match = fire_rule(subtree, rule)
                if match is not None:
                    return new_state, rule.label, match.path
            return subtree, None, None

    def _gillespie_step(self, subtree, interval):
        t = 0.0
        fired_any = False
        steps = 0
        while t < interval and steps < self.max_per_tick:
            candidates = self._enumerate_candidates(subtree)
            if not candidates:
                break
            total = sum(r for _, _, _, r in candidates)
            if total <= 0:
                break
            u = max(self.rng.random(), 1e-12)
            dt = -math.log(u) / total
            if t + dt > interval:
                break
            t += dt
            picked = self._pick_candidate(candidates)
            if picked is None:
                break
            rule, idx, match = picked
            subtree, _ = fire_rule(subtree, rule, match_index=idx)
            self.fired_log.append((t, rule.label, match.path))
            fired_any = True
            steps += 1
        return subtree, fired_any
