"""Tests for the port-compatibility Translator classifier.

Demonstrates that the bigraph-schema Translator kernel
(``register_translator`` / ``cross`` / ``Crossed`` / ``Refusal``) turns
silent process-bigraph wiring hazards into a classified ``Crossed`` or a loud
``Refusal`` — and, on a REAL composite, catches what ``Composite.initialize``
accepts in silence.
"""

from bigraph_schema.translator import Crossed, Refusal

from process_bigraph import allocate_core, Composite
from process_bigraph.composite import Process
from process_bigraph.wiring_translator import (
    PortCrossing,
    classify_port_crossing,
    iter_wired_output_crossings,
)


def _core():
    core = allocate_core()
    # Confirm process-bigraph's core inherits the kernel from bigraph-schema.
    assert hasattr(core, 'register_translator')
    assert hasattr(core, 'cross')
    return core


# ---------------------------------------------------------------------------
# 1. exact
# ---------------------------------------------------------------------------

def test_exact_float_to_float():
    core = _core()
    result = classify_port_crossing(core, 'float', 'float')
    assert isinstance(result, Crossed)
    assert isinstance(result.value, PortCrossing)
    assert result.value.tag == 'exact'
    assert result.value.join == 'float'


# ---------------------------------------------------------------------------
# 2. widened (integer -> float): the classifier flags what resolve hides
# ---------------------------------------------------------------------------

def test_widened_integer_to_float_is_flagged_not_silent():
    core = _core()

    # The silent status quo: plain resolve returns `float` with NO signal that
    # the producer's integer was widened.
    silent = core.resolve('integer', 'float')
    assert core.render(silent) == 'float'  # no widening signal whatsoever

    # The classifier turns that silent join into a declared, tagged crossing.
    result = classify_port_crossing(core, 'integer', 'float')
    assert isinstance(result, Crossed)
    assert isinstance(result.value, PortCrossing)
    assert result.value.tag == 'widened'
    assert result.value.join == 'float'         # names the join resolve took silently
    assert result.value.source == 'integer'
    assert 'widen' in result.value.note


# ---------------------------------------------------------------------------
# 3. semantics-shifted (overwrite[float] -> float): dropped Overwrite
# ---------------------------------------------------------------------------

def test_overwrite_into_additive_float_is_refused():
    core = _core()
    result = classify_port_crossing(core, 'overwrite[float]', 'float')
    assert isinstance(result, Refusal)
    assert 'semantics_shifted' in result.reason
    # The refusal names the exact hazard: the Overwrite wrapper is dropped and
    # apply falls back to additive Float.
    assert 'Overwrite' in result.reason
    assert 'additive' in result.reason


def test_overwrite_to_overwrite_is_exact_not_refused():
    # Control: identical wrappers on both sides must NOT be a semantics shift.
    core = _core()
    result = classify_port_crossing(core, 'overwrite[float]', 'overwrite[float]')
    assert isinstance(result, Crossed)
    assert result.value.tag == 'exact'


# ---------------------------------------------------------------------------
# 4. irreconcilable (map[float] -> float): resolve raises
# ---------------------------------------------------------------------------

def test_irreconcilable_map_into_float_is_refused():
    core = _core()

    # Ground the classification: this pair really does make resolve raise.
    raised = False
    try:
        core.resolve('map[float]', 'float')
    except Exception as e:
        raised = True
        assert 'cannot resolve types' in str(e)
    assert raised

    result = classify_port_crossing(core, 'map[float]', 'float')
    assert isinstance(result, Refusal)
    assert 'irreconcilable' in result.reason


def test_string_into_float_is_widened_not_irreconcilable():
    # Contrast documented in the task: string -> float does NOT raise; resolve
    # silently widens it. The classifier still flags it (widened), so the
    # silent pass becomes a signal — but it is not irreconcilable.
    core = _core()
    assert core.render(core.resolve('string', 'float')) == 'float'  # silent
    result = classify_port_crossing(core, 'string', 'float')
    assert isinstance(result, Crossed)
    assert result.value.tag == 'widened'


# ---------------------------------------------------------------------------
# 5. REAL composite: initialize accepts silently; classifier catches it
# ---------------------------------------------------------------------------

class ProducerInt(Process):
    """Emits an `integer` on its output port."""

    def inputs(self):
        return {}

    def outputs(self):
        return {'out': 'integer'}

    def update(self, state, interval):
        return {'out': 3}


class ProducerOverwrite(Process):
    """Emits an `overwrite[float]` (replacement) on its output port."""

    def inputs(self):
        return {}

    def outputs(self):
        return {'out': 'overwrite[float]'}

    def update(self, state, interval):
        return {'out': 7.0}


def test_real_composite_widened_and_semantics_shift_caught():
    core = _core()
    core.register_link('ProducerInt', ProducerInt)
    core.register_link('ProducerOverwrite', ProducerOverwrite)

    # A real composite wiring an integer output AND an overwrite[float] output
    # onto plain additive `float` stores.
    composite = Composite({'state': {
        'producer_int': {
            'address': 'local:ProducerInt', 'config': {}, 'interval': 1.0,
            'inputs': {}, 'outputs': {'out': ['level']}},
        'producer_ow': {
            'address': 'local:ProducerOverwrite', 'config': {}, 'interval': 1.0,
            'inputs': {}, 'outputs': {'out': ['amount']}},
        'level': 0.0,    # float store fed by an integer producer -> widening
        'amount': 0.0,   # additive float store fed by overwrite[float] -> semantics shift
    }}, core=core)

    # initialize accepted BOTH hazardous wirings with no complaint: the stores
    # are plain floats, the integer/overwrite mismatch was silently absorbed.
    assert core.render(composite.schema['level']) == 'float'
    assert core.render(composite.schema['amount']) == 'float'

    instances = {
        'producer_int': ProducerInt({}, core=core),
        'producer_ow': ProducerOverwrite({}, core=core),
    }

    # Run every wired output port-pair through the classifier.
    outcomes = {}
    for name, port, store_path, ptype, ctype in iter_wired_output_crossings(
            composite, instances):
        outcomes[name] = classify_port_crossing(core, ptype, ctype)

    assert set(outcomes) == {'producer_int', 'producer_ow'}

    # integer -> float store: classifier flags the widening initialize hid.
    widened = outcomes['producer_int']
    assert isinstance(widened, Crossed)
    assert widened.value.tag == 'widened'
    assert widened.value.source == 'integer'
    assert widened.value.join == 'float'

    # overwrite[float] -> additive float store: classifier REFUSES (the dropped
    # Overwrite that silently turns replacement into additive apply).
    shifted = outcomes['producer_ow']
    assert isinstance(shifted, Refusal)
    assert 'semantics_shifted' in shifted.reason
