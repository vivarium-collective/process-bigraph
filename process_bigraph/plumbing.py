"""
Plumbing Steps — dataflow restructuring primitives.

Each Step here is pure: it reshapes incoming match-sets without side
effects. They exist so composite documents can express the dataflow
patterns that Nextflow channel operators provide (mix, collect, combine,
groupTuple, join) while still running natively in the process-bigraph
engine. The native implementations are authoritative; the Nextflow
renderer translates each plumbing Step into the corresponding channel
operator.

See doc/nextflow_composite_spec.md in vEcoli (section 2) for the design.
"""

import itertools
from typing import Any, Dict

from process_bigraph.composite import Step


class Mix(Step):
    """Concatenate multiple streams into one.

    Renders to Nextflow `.mix()`.
    """

    nextflow_operator = 'mix'

    def inputs(self):
        return {'streams': 'list[list]'}

    def outputs(self):
        return {'merged': 'list'}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {'merged': list(itertools.chain.from_iterable(state['streams']))}


class Collect(Step):
    """Gather a stream into a single list.

    Semantically identity at runtime — the stream is already a list.
    Exists so the renderer can emit Nextflow `.collect()` to change
    queue-channel semantics to value-channel semantics.
    """

    nextflow_operator = 'collect'

    def inputs(self):
        return {'stream': 'list'}

    def outputs(self):
        return {'collected': 'list'}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {'collected': list(state['stream'])}


class Combine(Step):
    """Cartesian product of two streams.

    Each pair is emitted as a 2-element list ``[a_item, b_item]``.
    Renders to Nextflow `.combine()`.
    """

    nextflow_operator = 'combine'

    def inputs(self):
        return {'a': 'list', 'b': 'list'}

    def outputs(self):
        return {'product': 'list'}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pairs = [[x, y] for x, y in itertools.product(state['a'], state['b'])]
        return {'product': pairs}


class GroupBy(Step):
    """Partition a stream by the value of a named field.

    Each item must be a dict. ``key_field`` selects the grouping key.
    Output is a map from key value to the list of items carrying it.
    Renders to Nextflow `.groupTuple()` on the specified key.
    """

    nextflow_operator = 'groupTuple'

    config_schema = {
        'key_field': 'string',
    }

    def inputs(self):
        return {'stream': 'list'}

    def outputs(self):
        return {'groups': 'map[list]'}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        key_field = self.config['key_field']
        groups: Dict[Any, list] = {}
        for item in state['stream']:
            key = item[key_field]
            groups.setdefault(key, []).append(item)
        return {'groups': groups}


class Join(Step):
    """Inner-join two streams on a tuple of shared key fields.

    Items must be dicts. ``on`` lists the field names to match on.
    On a match, the left and right dicts are merged (right wins on
    conflicts, matching Python's ``{**l, **r}`` convention).
    Renders to Nextflow `.join()`.
    """

    nextflow_operator = 'join'

    config_schema = {
        'on': 'list[string]',
    }

    def inputs(self):
        return {'left': 'list', 'right': 'list'}

    def outputs(self):
        return {'joined': 'list'}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        keys = tuple(self.config['on'])
        right_index: Dict[tuple, list] = {}
        for item in state['right']:
            k = tuple(item[field] for field in keys)
            right_index.setdefault(k, []).append(item)

        joined = []
        for left in state['left']:
            k = tuple(left[field] for field in keys)
            for right in right_index.get(k, ()):
                joined.append({**left, **right})
        return {'joined': joined}
