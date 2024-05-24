"""
====================================
Composite, Process, and Step classes
====================================
"""

import abc
import copy
import math
import types
import collections
from typing import Dict


from bigraph_schema import Edge, TypeSystem, get_path, establish_path, set_path, deep_merge
from bigraph_schema.registry import Registry, validate_merge, visit_method

from process_bigraph.protocols import local_lookup, local_lookup_module


# TODO: implement these
def apply_process(schema, current, update, core):
    process_schema = schema.copy()
    process_schema.pop('_apply')
    return core.apply(
        process_schema,
        current,
        update)


def check_process(schema, state, core):
    return 'instance' in state and isinstance(
        state['instance'],
        Edge)


def fold_visit(schema, state, method, values, core):
    visit = visit_method(
        schema,
        state,
        method,
        values,
        core)

    return visit


def divide_process(schema, state, values, core):
    # daughter_configs must have a config per daughter

    daughter_configs = values.get(
        'daughter_configs',
        [{} for index in range(values['divisions'])])

    if 'config' not in state:
        return daughter_configs

    existing_config = state['config']

    divisions = []
    for index in range(values['divisions']):
        daughter_config = copy.deepcopy(
            existing_config)
        daughter_config = deep_merge(
            daughter_config,
            daughter_configs[index])
        
        # TODO: provide a way to override inputs and outputs
        daughter_state = {
            'address': state['address'],
            'config': daughter_config,
            'inputs': copy.deepcopy(state['inputs']),
            'outputs': copy.deepcopy(state['outputs'])}

        if 'interval' in state:
            daughter_state['interval'] = state['interval']

        divisions.append(daughter_state)

    return divisions


def serialize_process(schema, value, core):
    """Serialize a process to a JSON-safe representation."""
    # TODO -- need to get back the protocol: address and the config
    process = value.copy()
    del process['instance']
    return process


def assert_interface(interface):
    required_keys = ['inputs', 'outputs']
    existing_keys = set(interface.keys())
    assert existing_keys == set(required_keys), f"every interface requires an inputs schema and an outputs schema, not {existing_keys}"


DEFAULT_INTERVAL = 1.0


TYPE_SCHEMAS = {
    'float': 'float'}


def deserialize_process(schema, encoded, core):
    """Deserialize a process from a serialized state.

    This function is used by the type system to deserialize a process.

    :param encoded: A JSON-safe representation of the process.
    :param bindings: The bindings to use for deserialization.
    :param core: The type system to use for deserialization.

    :returns: The deserialized state with an instantiated process.
    """
    deserialized = encoded.copy()
    if 'address' not in deserialized:
        return deserialized

    protocol, address = encoded['address'].split(':', 1)

    if 'instance' in deserialized:
        instantiate = type(deserialized['instance'])
    else:
        process_lookup = core.protocol_registry.access(protocol)
        if not process_lookup:
            raise Exception(f'protocol "{protocol}" not implemented')

        instantiate = process_lookup(core, address)
        if not instantiate:
            raise Exception(f'process "{address}" not found')

    config = core.deserialize(
        instantiate.config_schema,
        encoded.get('config', {}))

    interval = core.deserialize(
        'interval',
        encoded.get('interval'))

    if not 'instance' in deserialized:
        process = instantiate(config, core=core)
        deserialized['instance'] = process

    deserialized['config'] = config
    deserialized['interval'] = interval
    deserialized['_inputs'] = deserialized['instance'].inputs()
    deserialized['_outputs'] = deserialized['instance'].outputs()

    return deserialized


def deserialize_step(schema, encoded, core):
    deserialized = encoded.copy()
    protocol, address = encoded['address'].split(':', 1)

    if 'instance' in deserialized:
        instantiate = type(deserialized['instance'])
    else:
        process_lookup = core.protocol_registry.access(protocol)
        if not process_lookup:
            raise Exception(f'protocol "{protocol}" not implemented')

        instantiate = process_lookup(core, address)
        if not instantiate:
            raise Exception(f'process "{address}" not found')

    config = core.deserialize(
        instantiate.config_schema,
        encoded.get('config', {}))

    if not 'instance' in deserialized:
        process = instantiate(config, core=core)
        deserialized['instance'] = process

    deserialized['config'] = config
    deserialized['_inputs'] = deserialized['instance'].inputs()
    deserialized['_outputs'] = deserialized['instance'].outputs()

    return deserialized


process_types = {
    'protocol': {
        '_type': 'protocol',
        '_inherit': 'string'},

    'interval': {
        '_type': 'interval',
        '_inherit': 'float',
        '_apply': 'set',
        '_default': '1.0'},

    'step': {
        '_type': 'step',
        '_inherit': 'edge',
        '_apply': apply_process,
        '_serialize': serialize_process,
        '_deserialize': deserialize_step,
        '_check': check_process,
        '_fold': fold_visit,
        '_divide': divide_process,
        '_description': '',
        # TODO: support reference to type parameters from other states
        'address': 'protocol',
        'config': 'tree[any]'},

    'process': {
        '_type': 'process',
        '_inherit': 'edge',
        '_apply': apply_process,
        '_serialize': serialize_process,
        '_deserialize': deserialize_process,
        '_check': check_process,
        '_fold': fold_visit,
        '_divide': divide_process,
        '_description': '',
        # TODO: support reference to type parameters from other states
        'interval': 'interval',
        'address': 'protocol',
        'config': 'tree[any]'},
}


def register_process_types(core):
    for process_key, process_type in process_types.items():
        core.register(process_key, process_type)

    return core


def register_protocols(core):
    core.protocol_registry.register('local', local_lookup)


def register_emitters(core):
    core.register_process('console-emitter', ConsoleEmitter)
    core.register_process('ram-emitter', RAMEmitter)


class ProcessTypes(TypeSystem):
    def __init__(self):
        super().__init__()
        self.process_registry = Registry()
        self.protocol_registry = Registry()

        register_process_types(self)
        register_protocols(self)
        register_emitters(self)

        self.register_process('composite', Composite)


    def register_process(self, name, process_data):
        self.process_registry.register(name, process_data)


    def infer_schema(self, schema, state, top_state=None, path=None):
        """
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (processes/steps) during this process
        """

        schema = schema or {}
        # TODO: deal with this
        if schema == '{}':
            schema = {}

        top_state = top_state or state
        path = path or ()

        if isinstance(state, dict):
            if '_type' in state:
                state_type = {
                    key: value
                    for key, value in state.items()
                    if key.startswith('_')}

                state_schema = self.access(
                    state_type)

                hydrated_state = self.deserialize(
                    state_schema,
                    state)

                schema, top_state = self.set_slice(
                    schema,
                    top_state,
                    path,
                    state_schema,
                    hydrated_state)

                # TODO: fix is_descendant
                # if core.type_registry.is_descendant('process', state_schema) or core.registry.is_descendant('step', state_schema):
                if state_type['_type'] == 'process' or state_type['_type'] == 'step':
                    port_schema = hydrated_state['instance'].interface()
                    assert_interface(
                        port_schema)

                    for port_key in ['inputs', 'outputs']:
                        subschema = port_schema.get(
                            port_key, {})

                        schema = set_path(
                            schema,
                            path + (f'_{port_key}',),
                            subschema)

                        ports = hydrated_state.get(
                            port_key, {})

                        schema = self.infer_wires(
                            subschema,
                            hydrated_state,
                            ports,
                            top_schema=schema,
                            path=path)

            elif '_type' in schema:
                hydrated_state = self.deserialize(schema, state)
                top_state = set_path(
                    top_state,
                    path,
                    hydrated_state)
            else:
                for key, value in state.items():
                    inner_path = path + (key,)
                    if get_path(schema, inner_path) is None or get_path(state, inner_path) is None or (
                            isinstance(value, dict) and '_type' in value):
                        schema, top_state = self.infer_schema(
                            schema,
                            value,
                            top_state=top_state,
                            path=inner_path)

        elif isinstance(state, str):
            pass

        else:
            schema, top_state = super().infer_schema(
                schema,
                state,
                top_state,
                path)

        return schema, top_state

    def infer_edge(self, schema, wires):
        schema = schema or {}
        edge = {}

        if isinstance(wires, str):
            import ipdb; ipdb.set_trace()

        for port_key, wire in wires.items():
            if isinstance(wire, dict):
                edge[port_key] = self.infer_edge(
                    schema.get(port_key, {}),
                    wire)
            else:
                subschema = get_path(schema, wire)
                edge[port_key] = subschema

        return edge

    def initialize_edge_state(self, schema, path, edge):
        initial_state = edge['instance'].initial_state()
        if not initial_state:
            return initial_state

        input_ports = get_path(schema, path + ('_inputs',))
        output_ports = get_path(schema, path + ('_outputs',))
        ports = {
            '_inputs': input_ports,
            '_outputs': output_ports}

        input_state = {}
        if input_ports:
            input_state = self.project_edge(
                ports,
                edge,
                path[:-1],
                initial_state,
                ports_key='inputs')

        output_state = {}
        if output_ports:
            output_state = self.project_edge(
                ports,
                edge,
                path[:-1],
                initial_state,
                ports_key='outputs')

        state = deep_merge(input_state, output_state)

        return state


    def lookup_address(self, address):
        protocol, config = address.split(':')

        if protocol == 'local':
            self.lookup_local(config)


def hierarchy_depth(hierarchy, path=()):
    """
    Create a mapping of every path in the hierarchy to the node living at
    that path in the hierarchy.
    """

    base = {}

    for key, inner in hierarchy.items():
        down = tuple(path + (key,))
        if key.startswith('_'):
            base[path] = inner
        elif isinstance(inner, dict) and 'instance' not in inner:
            base.update(hierarchy_depth(inner, down))
        else:
            base[down] = inner

    return base


# deal with steps vs temporal process vs edges

class SyncUpdate():
    def __init__(self, update):
        self.update = update

    def get(self):
        return self.update


class Step(Edge):
    """Step base class."""
    # TODO: support trigger every time
    #   as well as dependency trigger
    config_schema = {}


    def __init__(self, config=None, core=None):
        self.core = core or ProcessTypes()

        if config is None:
            config = {}

        self.config = self.core.fill(
            self.config_schema,
            config)


    def initial_state(self):
        return {}


    def invoke(self, state, _=None):
        update = self.update(state)
        sync = SyncUpdate(update)
        return sync


    def update(self, state):
        return {}


class Process(Edge):
    """Process parent class.

      All :term:`process` classes must inherit from this class. Each
      class can provide a ``defaults`` class variable to specify the
      process defaults as a dictionary.

      Note that subclasses should call the superclass init function
      first. This allows the superclass to correctly save the initial
      parameters before they are mutated by subclass constructor code.
      We need access to the original parameters for serialization to
      work properly.

      Args:
          config: Override the class defaults. This dictionary may
              also contain the following special keys (TODO):
    """
    config_schema = {}

    def __init__(self, config=None, core=None):
        self.core = core or ProcessTypes()

        if config is None:
            config = {}

        # check that all keywords in config are in config_schema
        for key in config.keys():
            if key not in self.config_schema:
                raise Exception(f'config key {key} not in config_schema for {self.__class__.__name__}')

        # fill in defaults for config
        self.config = self.core.fill(
            self.config_schema,
            config)


    def initial_state(self):
        return {}


    def invoke(self, state, interval):
        update = self.update(state, interval)
        sync = SyncUpdate(update)
        return sync


    def update(self, state, interval):
        return {}


    # TODO: should we include run(interval) here?
    #   process would have to maintain state


class Defer:
    """Allows for delayed application of a function to an update.

    The object simply holds the provided arguments until it's time
    for the computation to be performed. Then, the function is
    called.

    Args:
        defer: An object with a ``.get_command_result()`` method
            whose output will be passed to the function. For
            example, the object could be an
            :py:class:`vivarium.core.process.Process` object whose
            ``.get_command_result()`` method will return the process
            update.
        function: The function. For example,
            :py:func:`invert_topology` to transform the returned
            update.
        args: Passed as the second argument to the function.
    """

    def __init__(
            self,
            defer,
            f,
            args
    ):

        self.defer = defer
        self.f = f
        self.args = args


    def get(self):
        """Perform the deferred computation.

        Returns:
            The result of calling the function.
        """
        return self.f(
            self.defer.get(),
            self.args)


def find_instances(state, instance_type='process_bigraph.composite.Process'):
    process_class = local_lookup_module(instance_type)
    found = {}

    for key, inner in state.items():
        if isinstance(inner, dict):
            if isinstance(inner.get('instance'), process_class):
                found[key] = inner
            elif not key.startswith('_'):
                inner_instances = find_instances(
                    inner,
                    instance_type=instance_type)

                if inner_instances:
                    found[key] = inner_instances
    return found


def find_instance_paths(state, instance_type='process_bigraph.composite.Process'):
    instances = find_instances(state, instance_type)
    return hierarchy_depth(instances)


def find_step_triggers(path, step):
    prefix = tuple(path[:-1])
    triggers = {}
    wire_paths = find_leaves(
        step['inputs'])

    for wire in wire_paths:
        trigger_path = tuple(prefix) + tuple(wire)
        if trigger_path not in triggers:
            triggers[trigger_path] = []
        triggers[trigger_path].append(path)

    return triggers


def explode_path(path):
    explode = ()
    paths = [explode]

    for node in path:
        explode = explode + (node,)
        paths.append(explode)

    return paths


def merge_collections(existing, new):
    if existing is None:
        existing = {}
    if new is None:
        new = {}
    for key, value in new.items():
        if key in existing:
            if isinstance(existing[key], dict) and isinstance(new[key], collections.abc.Mapping):
                merge_collections(existing[key], new[key])
            elif isinstance(existing[key], list) and isinstance(new[key], collections.abc.Sequence):
                existing[key].extend(new[key])
            else:
                raise Exception(
                    f'cannot merge collections as they do not match:\n{existing}\n{new}')
        else:
            existing[key] = value

    return existing


def empty_front(time):
    return {
        'time': time,
        'update': {}
    }


def find_leaves(d, path=None):
    leaves = []
    path = ()

    if isinstance(d, list):
        leaves = d
    elif isinstance(d, tuple):
        leaves.append(d)
    else:
        for key, value in d.items():
            if isinstance(value, dict):
                subleaves = find_leaves(value, path + (key,))
                leaves.extend(subleaves)
            else:
                leaves.append(path + tuple(value))

    return leaves


def build_step_network(steps):
    ancestors = {
        step_key: {
            'input_paths': None,
            'output_paths': None}
        for step_key in steps}

    nodes = {}

    for step_key, step in steps.items():
        for other_key, other_step in steps.items():
            if step_key == other_key:
                continue

            schema = step['instance'].interface()
            other_schema = other_step['instance'].interface()

            assert_interface(schema)
            assert_interface(other_schema)

            if ancestors[step_key]['input_paths'] is None:
                ancestors[step_key]['input_paths'] = find_leaves(
                    step['inputs'])
            input_paths = ancestors[step_key]['input_paths']

            if ancestors[step_key]['output_paths'] is None:
                ancestors[step_key]['output_paths'] = find_leaves(
                    step.get('outputs', {}))
            output_paths = ancestors[step_key]['output_paths']

            for input in input_paths:
                path = tuple(input)
                if not path in nodes:
                    nodes[path] = {
                        'before': set([]),
                        'after': set([])}
                nodes[path]['after'].add(step_key)

            for output in output_paths:
                path = tuple(output)
                if not path in nodes:
                    nodes[path] = {
                        'before': set([]),
                        'after': set([])}
                nodes[path]['before'].add(step_key)

    return ancestors, nodes


def combined_step_network(steps):
    steps, nodes = build_step_network(steps)

    trigger_state = {
        'steps': steps,
        'nodes': nodes}

    return trigger_state


def build_trigger_state(nodes):
    return {
        key: value['before'].copy()
        for key, value in nodes.items()}


def find_downstream(steps, nodes, upstream):
    downstream = set(upstream)
    visited = set([])
    previous_len = -1

    while len(downstream) > len(visited) and len(visited) > previous_len:
        previous_len = len(visited)
        down = set([])
        for step_path in downstream:
            if step_path not in visited:
                step_outputs = steps[step_path]['output_paths']
                if step_outputs is None:
                    step_outputs = []  # Ensure step_outputs is always an iterable
                for output in step_outputs:
                    for dependent in nodes[output]['after']:
                        down.add(dependent)
                visited.add(step_path)
        downstream |= down

    return downstream


def determine_steps(steps, remaining, fulfilled):
    to_run = []
    for step_path in remaining:
        step_inputs = steps[step_path]['input_paths']
        if step_inputs is None:
            step_inputs = []
        all_fulfilled = True
        for input in step_inputs:
            if len(fulfilled[input]) > 0:
                all_fulfilled = False
        if all_fulfilled:
            to_run.append(step_path)

    for step_path in to_run:
        remaining.remove(step_path)
        step_outputs = steps[step_path]['output_paths']
        if step_outputs is None:
            step_outputs = []
        for output in step_outputs:
            fulfilled[output].remove(step_path)

    return to_run, remaining, fulfilled


def interval_time_precision(timestep):
    # get number of decimal places to set global time precision
    timestep_str = str(timestep)
    global_time_precision = 0
    if '.' in timestep_str:
        _, decimals = timestep_str.split('.')
        global_time_precision = len(decimals)

    return global_time_precision


class Composite(Process):
    """
    Composite parent class.
    """


    config_schema = {
        'composition': 'schema',
        'state': 'tree[any]',
        'interface': {
            'inputs': 'schema',
            'outputs': 'schema'},
        'bridge': {
            'inputs': 'wires',
            'outputs': 'wires'},
        'global_time_precision': 'maybe[float]'}


    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # insert global_time into schema if not present
        initial_composition = self.config.get('composition', {})
        if 'global_time' not in initial_composition:
            initial_composition['global_time'] = 'float'

        # insert global_time into state if not present
        initial_state = self.config.get('state', {})
        if 'global_time' not in initial_state:
            initial_state['global_time'] = 0.0

        composition, state = self.core.complete(
            initial_composition,
            initial_state)

        self.composition = copy.deepcopy(
            self.core.access(composition))

        # TODO: add flag to self.core.access(copy=True)
        self.bridge = self.config.get('bridge', {})

        self.find_instance_paths(
            state)

        # merge the processes and steps into a single "edges" dict
        self.edge_paths = self.process_paths.copy()
        self.edge_paths.update(self.step_paths)

        # get the initial_state() for each edge and merge
        # them all together, validating that there are no
        # contradictions in the state (paths from initial_state
        # that conflict/have different values at the same path)
        edge_state = {}
        for path, edge in self.edge_paths.items():
            initial = self.core.initialize_edge_state(
                self.composition,
                path,
                edge)

            try:
                edge_state = deep_merge(edge_state, initial)
            except:
                raise Exception(
                    f'initial state from edge does not match initial state from other edges:\n{path}\n{edge}\n{edge_state}')

        state = deep_merge(edge_state, state)

        self.state = self.core.deserialize(
            self.composition,
            state)

        # TODO: call validate on this composite, not just check
        # assert self.core.validate(
        #     self.composition,
        #     self.state)

        self.process_schema = {}
        for port in ['inputs', 'outputs']:
            self.process_schema[port] = self.core.infer_edge(
                self.composition,
                self.bridge[port])

        self.global_time_precision = self.config[
            'global_time_precision']

        self.step_triggers = {}

        for step_path, step in self.step_paths.items():
            step_triggers = find_step_triggers(
                step_path, step)
            self.step_triggers = merge_collections(
                self.step_triggers,
                step_triggers)

        self.steps_run = set([])

        self.front: Dict = {
            path: empty_front(self.state['global_time'])
            for path in self.process_paths}

        self.bridge_updates = []

        self.step_dependencies, self.node_dependencies = build_step_network(
            self.step_paths)

        self.reset_step_state(self.step_paths)
        self.to_run = self.cycle_step_state()

        # self.run_steps(self.to_run)


    def reset_step_state(self, step_paths):
        self.trigger_state = build_trigger_state(
            self.node_dependencies)

        self.steps_remaining = set(step_paths)


    def cycle_step_state(self):
        to_run, self.steps_remaining, self.trigger_state = determine_steps(
            self.step_dependencies,
            self.steps_remaining,
            self.trigger_state)

        return to_run


    def find_instance_paths(self, state):
        # find all processes, steps, and emitter in the state
        self.process_paths = find_instance_paths(
            state,
            'process_bigraph.composite.Process')

        self.step_paths = find_instance_paths(
            state,
            'process_bigraph.composite.Step')

        self.emitter_paths = find_instance_paths(
            state,
            'process_bigraph.composite.Emitter')


    def inputs(self):
        return self.process_schema.get('inputs', {})


    def outputs(self):
        return self.process_schema.get('outputs', {})


    def merge(self, initial_state):
        self.state = self.core.merge(
            self.composition,
            self.state,
            initial_state)


    def process_update(
            self,
            path,
            process,
            states,
            interval,
            ports_key='outputs'):

        """Start generating a process's update.

        This function is similar to :py:meth:`_invoke_process` except in
        addition to triggering the computation of the process's update
        (by calling ``_invoke_process``), it also generates a
        :py:class:`Defer` object to transform the update into absolute
        terms.

        Args:
            path: Path to process.
            process: The process.
            states: Simulation state to pass to process's
                ``next_update`` method.
            interval: Interval for which to compute the update.

        Returns:
            Tuple of the deferred update (in absolute terms) and
            ``store``.
        """
        update = process['instance'].invoke(states, interval)

        def defer_project(update, args):
            schema, state, path = args
            return self.core.project_edge(
                schema,
                state,
                path,
                update,
                ports_key)

        absolute = Defer(
            update,
            defer_project, (
                self.composition,
                self.state,
                path))

        return absolute


    def run_process(self, path, process, end_time, full_step, force_complete):
        if path not in self.front:
            self.front[path] = empty_front(
                self.state['global_time'])

        process_time = self.front[path]['time']
        if process_time <= self.state['global_time']:
            if self.front[path].get('future'):
                future_front = self.front[path]['future']
                process_interval = future_front['interval']
                store = future_front['store']
                state = future_front['state']
                del self.front[path]['future']
            else:
                state = self.core.view_edge(
                    self.composition,
                    self.state,
                    path)

                process_interval = process['interval']

            if force_complete:
                # force the process to complete at end_time
                future = min(process_time + process_interval, end_time)
            else:
                future = process_time + process_interval

            if self.global_time_precision is not None:
                # set future time based on global_time_precision
                future = round(future, self.global_time_precision)

            # absolute interval
            interval = future - self.state['global_time']
            if interval < full_step:
                full_step = interval

            if future <= end_time:
                update = self.process_update(
                    path,
                    process,
                    state,
                    process_interval
                )

                # update front, to be applied at its projected time
                self.front[path]['time'] = future
                self.front[path]['update'] = update

        else:
            # don't shoot past processes that didn't run this time
            process_delay = process_time - self.state['global_time']
            if process_delay < full_step:
                full_step = process_delay

        return full_step


    def apply_updates(self, updates):
        # view_expire = False
        update_paths = []

        for defer in updates:
            series = defer.get()
            if series is None:
                continue

            if not isinstance(series, list):
                series = [series]

            for update in series:
                paths = hierarchy_depth(update)
                update_paths.extend(paths.keys())

                self.state = self.core.apply_update(
                    self.composition,
                    self.state,
                    update)

                bridge_update = self.core.view(
                    self.interface()['outputs'],
                    self.bridge['outputs'],
                    (),
                    update)

                if bridge_update:
                    self.bridge_updates.append(bridge_update)

        return update_paths

                # view_expire_update = self.apply_update(up, store)
                # view_expire = view_expire or view_expire_update

        # if view_expire:
        #     self.state.build_topology_views()


    def expire_process_paths(self, update_paths):
        for update_path in update_paths:
            for process_path in self.process_paths.copy():
                updated = all([
                    update == process
                    for update, process in zip(update_path, process_path)])

                if updated:
                    self.find_instance_paths(
                        self.state)
                    return

                    # del self.process_paths[process_path]

                    # target_schema, target_state = self.core.slice(
                    #     self.composition,
                    #     self.state,
                    #     update_path)

                    # process_subpaths = find_instance_paths(
                    #     target_state,
                    #     'process_bigraph.composite.Process')

                    # for subpath, process in process_subpaths.items():
                    #     process_path = update_path + subpath
                    #     self.process_paths[process_path] = process


    def run(self, interval, force_complete=False):
        if self.to_run:
            self.run_steps(self.to_run)
            self.to_run = None

        end_time = self.state['global_time'] + interval
        while self.state['global_time'] < end_time or force_complete:
            full_step = math.inf

            for path in self.process_paths:
                process = get_path(self.state, path)
                full_step = self.run_process(
                    path,
                    process,
                    end_time,
                    full_step,
                    force_complete)

            # apply updates based on process times in self.front
            if full_step == math.inf:
                # no processes ran, jump to next process
                next_event = end_time
                for path in self.front.keys():
                    if self.front[path]['time'] < next_event:
                        next_event = self.front[path]['time']
                self.state['global_time'] = next_event

            elif self.state['global_time'] + full_step <= end_time:
                # at least one process ran within the interval
                # increase the time, apply updates, and continue
                self.state['global_time'] += full_step

                # apply updates that are behind global time
                updates = []
                paths = []
                for path, advance in self.front.items():
                    if advance['time'] <= self.state['global_time'] \
                            and advance['update']:
                        new_update = advance['update']
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                # get all update paths, then trigger steps that
                # depend on those paths
                update_paths = self.apply_updates(updates)
                self.expire_process_paths(update_paths)
                self.trigger_steps(update_paths)

                # # display and emit
                # if self.progress_bar:
                #     print_progress_bar(self.global_time, end_time)
                # if self.emit_step == 1:
                #     self._emit_store_data()
                # elif emit_time <= self.global_time:
                #     while emit_time <= self.global_time:
                #         self._emit_store_data()
                #         emit_time += self.emit_step

            else:
                # all processes have run past the interval
                self.state['global_time'] = end_time

            if force_complete and self.state['global_time'] == end_time:
                force_complete = False


    def determine_steps(self):
        to_run = []
        for step_key, wires in trigger_state['steps']:
            fulfilled = True
            for input in wires['input_paths']:
                if len(trigger_state['states'][tuple(input)]) > 0:
                    fulfilled = False
                    break
            if fulfilled:
                to_run.append(step_key)

        for step_key in to_run:
            wires = trigger_state['steps'][step_key]
            for output in wires['output_paths']:
                trigger_state['states'][tuple(output)].remove(step_key)

        return to_run, trigger_state


    def run_steps(self, step_paths):
        if len(step_paths) > 0:
            updates = []
            for step_path in step_paths:
                step = get_path(
                    self.state,
                    step_path)

                state = self.core.view_edge(
                    self.composition,
                    self.state,
                    step_path,
                    'inputs')

                step_update = self.process_update(
                    step_path,
                    step,
                    state,
                    -1.0,
                    'outputs')

                updates.append(step_update)

            update_paths = self.apply_updates(updates)
            to_run = self.cycle_step_state()
            if len(to_run) > 0:
                self.run_steps(to_run)
            else:
                self.steps_run = set([])
        else:
            self.steps_run = set([])


    def trigger_steps(self, update_paths):
        steps_to_run = []

        for update_path in update_paths:
            paths = explode_path(update_path)
            for path in paths:
                step_paths = self.step_triggers.get(path, [])
                for step_path in step_paths:
                    if step_path is not None and step_path not in self.steps_run:
                        steps_to_run.append(step_path)
                        self.steps_run.add(step_path)

        steps_to_run = find_downstream(
            self.step_dependencies,
            self.node_dependencies,
            steps_to_run)

        self.reset_step_state(steps_to_run)
        to_run = self.cycle_step_state()

        self.run_steps(to_run)


    def emit_port(self, emitter_path, process_path, port_path):
        pass


    def gather_results(self, queries=None):
        '''
        a map of paths to emitter --> queries for the emitter at that path
        '''

        if queries is None:
            queries = {
                path: None
                for path in self.emitter_paths.keys()}

        results = {}
        for path, query in queries.items():
            emitter = get_path(self.state, path)
            results[path] = emitter['instance'].query(query)

        return results

    def update(self, state, interval):
        # do everything

        projection = self.core.project(
            self.interface()['inputs'],
            self.bridge['inputs'],
            [],
            state)

        self.merge(
            projection)

        self.run(interval)

        updates = self.bridge_updates
        self.bridge_updates = []

        return updates


class Emitter(Step):
    """Base emitter class. An `Emitter` implementation instance diverts all querying of data to
        the primary historical collection whose type pertains to Emitter child, i.e:
            database-emitter=>`pymongo.Collection`, ram-emitter=>`.RamEmitter.history`(`List`)
    """
    config_schema = {
        'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

    def query(self, query=None):
        return {}

    def update(self, state) -> Dict:
        return {}


class ConsoleEmitter(Emitter):

    def update(self, state) -> Dict:
        print(state)
        return {}


class RAMEmitter(Emitter):

    def __init__(self, config, core):
        super().__init__(config, core)
        self.history = []


    def update(self, state) -> Dict:
        self.history.append(copy.deepcopy(state))
        return {}


    def query(self, query=None):
        if isinstance(query, list):
            result = {}
            for path in query:
                element = get_path(self.history, path)
                result = set_path(result, path, element)
        else:
            result = self.history

        return result


# def test_emitter():
#     composite = Composite({})

#     composite.add_emitter(['emitters', 'ram'], 'ram-emitter')
#     composite.emit_port(
#         ['emitters', 'ram'],
#         ['processes', 'translation'],
#         ['outputs', 'protein'])
