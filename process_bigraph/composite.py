"""
====================================
Composite, Process, and Step classes
====================================
"""

import abc
import copy
import math
import collections
from typing import Dict
from bigraph_schema.registry import deep_merge, validate_merge, get_path
from process_bigraph.type_system import types
from process_bigraph.protocols import local_lookup_module


def hierarchy_depth(hierarchy, path=()):
    """
    Create a mapping of every path in the hierarchy to the node living at
    that path in the hierarchy.
    """

    base = {}

    for key, inner in hierarchy.items():
        down = tuple(path + (key,))
        if isinstance(inner, dict) and 'instance' not in inner:
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


# TODO: create base class for Step and Process
#   maybe it comes from bigraph-schema?
class Edge:
    def __init__(self):
        pass


class Step(Edge):
    """Step base class."""
    # TODO: support trigger every time
    #   as well as dependency trigger
    config_schema = {}

    def __init__(self, config=None, local_types=None):
        self.types = local_types or types

        if config is None:
            config = {}

        self.config = self.types.fill(
            self.config_schema,
            config)

    def schema(self):
        return {}

    def initial_state(self):
        return {}
        # initial = {}
        # return types.fill(
        #     self.schema(),
        #     initial)


    def project_state(self, ports, wires, path, state):
        inputs = {}
        if 'inputs' in ports and 'inputs' in wires:
            inputs = self.types.project(
                ports['inputs'],
                wires['inputs'],
                path,
                state)

        outputs = {}
        if 'outputs' in ports and 'outputs' in wires:
            outputs = self.types.project(
                ports['outputs'],
                wires['outputs'],
                path,
                state)

        result = deep_merge(inputs, outputs)
        
        return result


    def invoke(self, state, _=None):
        update = self.update(state)
        sync = SyncUpdate(update)
        return sync

    @abc.abstractmethod
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

    def __init__(self, config=None, local_types=None):
        self.types = local_types or types
        if config is None:
            config = {}

        self.config = self.types.fill(
            self.config_schema,
            config)


    @abc.abstractmethod
    def schema(self):
        return {}


    def initial_state(self):
        return {}
        # initial = {}
        # return types.fill(
        #     self.schema(),
        #     initial)


    def project_state(self, ports, wires, path, state):
        return self.types.project(
            ports,
            wires,
            path,
            state)


    def invoke(self, state, interval):
        update = self.update(state, interval)
        sync = SyncUpdate(update)
        return sync


    @abc.abstractmethod
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

# TODO maybe keep wires as tuples/paths to distinguish them from schemas?


def find_instances(state, instance_type='process_bigraph.composite.Process'):
    process_class = local_lookup_module(instance_type)
    found = {}

    for key, inner in state.items():
        if isinstance(inner, dict) and isinstance(inner.get('instance'), process_class):
            found[key] = inner

    return found


def find_processes(state):
    return find_instances(state, 'process_bigraph.composite.Process')


def find_steps(state):
    return find_instances(state, 'process_bigraph.composite.Step')


def find_instance_paths(state, instance_type='process_bigraph.composite.Process'):
    instances = find_instances(state, instance_type)
    return hierarchy_depth(instances)


def find_step_triggers(path, step):
    prefix = tuple(path[:-1])
    triggers = {}
    wire_paths = find_leaves(
        step['wires']['inputs'])

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
            # 'ancestors': [],
            'input_paths': None,
            'output_paths': None}
        for step_key in steps}

    nodes = {}

    for step_key, step in steps.items():
        for other_key, other_step in steps.items():
            if step_key == other_key:
                continue

            schema = step['instance'].schema()
            wires = step['wires']
            other_schema = other_step['instance'].schema()
            other_wires = other_step['wires']

            if ancestors[step_key]['input_paths'] is None:
                ancestors[step_key]['input_paths'] = find_leaves(
                    wires['inputs'])
            input_paths = ancestors[step_key]['input_paths']

            if ancestors[step_key]['output_paths'] is None:
                ancestors[step_key]['output_paths'] = find_leaves(
                    wires.get('outputs', {}))
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
                for output in steps[step_path]['output_paths']:
                    for dependent in nodes[output]['after']:
                        down.add(dependent)
                visited.add(step_path)
        downstream ^= down

    return downstream


def determine_steps(steps, remaining, fulfilled):
    to_run = []

    for step_path in remaining:
        step_inputs = steps[step_path]['input_paths']
        all_fulfilled = True
        for input in step_inputs:
            if len(fulfilled[input]) > 0:
                all_fulfilled = False
        if all_fulfilled:
            to_run.append(step_path)

    for step_path in to_run:
        remaining.remove(step_path)
        step_outputs = steps[step_path]['output_paths']
        for output in step_outputs:
            fulfilled[output].remove(step_path)

    return to_run, remaining, fulfilled


class Composite(Process):
    """
    Composite parent class.
    """


    config_schema = {
        # TODO: add schema type
        'composition': 'tree[any]',
        'state': 'tree[any]',
        'schema': 'tree[any]',
        'bridge': 'wires',
        'global_time_precision': 'maybe[float]'}


    # TODO: if processes are serialized, deserialize them first
    def __init__(self, config=None, local_types=None):
        super().__init__(config, local_types)

        initial_composition = self.config.get('composition', {})
        if 'global_time' not in initial_composition:
            initial_composition['global_time'] = 'float'
        initial_composition = types.access(
            initial_composition)

        initial_state = self.config.get('state', {})
        if 'global_time' not in initial_state:
            initial_state['global_time'] = 0.0

        initial_state = types.hydrate(
            initial_composition,
            initial_state)

        initial_schema = types.access(
            self.config.get('schema', {})) or {}
        self.bridge = self.config.get('bridge', {})

        # fill in the parts of the composition schema
        # determined by the state
        composition, state = types.infer_schema(
            initial_composition,
            initial_state)
        # TODO: add flag to types.access(copy=True)
        composition_schema = types.access(composition)
        self.composition = copy.deepcopy(composition_schema)

        # find all processes, steps, and emitters in the state
        self.process_paths = find_instance_paths(
            state,
            'process_bigraph.composite.Process')

        self.step_paths = find_instance_paths(
            state,
            'process_bigraph.composite.Step')

        self.emitter_paths = find_instance_paths(
            state,
            'process_bigraph.emitter.Emitter')

        # merge the processes and steps into a single "edges" dict
        self.edge_paths = self.process_paths.copy()
        self.edge_paths.update(self.step_paths)

        # get the initial_state() for each edge and merge
        # them all together, validating that there are no
        # contradictions in the state (paths from initial_state
        # that conflict/have different values at the same path)
        edge_state = {}
        for path, edge in self.edge_paths.items():
            initial = types.initialize_edge_state(
                self.composition,
                path,
                edge)

            try:
                edge_state = validate_merge(state, edge_state, initial)
            except:
                raise Exception(
                    f'initial state from edge does not match initial state from other edges:\n{path}\n{edge}\n{edge_state}')

        state = deep_merge(edge_state, state)

        # calling hydrate here assumes all processes have already been
        # deserialized in the call to infer_schema above.
        self.state = types.hydrate(
            self.composition,
            state)

        self.process_schema = types.infer_edge(
            self.composition,
            self.bridge)

        self.global_time_precision = self.config['global_time_precision']

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
        to_run = self.cycle_step_state()

        self.run_steps(to_run)


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


    def schema(self):
        return self.process_schema


    def process_update(
            self,
            path,
            process,
            states,
            interval,
            ports_key=None,
    ):
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
            return types.project_edge(
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
            self.front[path] = empty_front(self.state['global_time'])
        process_time = self.front[path]['time']
        if process_time <= self.state['global_time']:
            if self.front[path].get('future'):
                future_front = self.front[path]['future']
                process_interval = future_front['interval']
                store = future_front['store']
                state = future_front['state']
                del self.front[path]['future']
            else:
                state = types.view_edge(
                    self.composition,
                    self.state,
                    path)

                process_interval = process['interval']
                # process_timestep = process['instance'].calculate_timestep(state)

            if force_complete:
                # force the process to complete at end_time
                future = min(process_time + process_interval, end_time)
            else:
                future = process_time + process_interval

            if self.global_time_precision is not None:
                # set future time based on global_time_precision
                future = round(future, self.global_time_precision)

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

                # absolute interval
                interval = future - self.state['global_time']
                if interval < full_step:
                    full_step = interval
            else:
                # absolute interval
                interval = future - self.state['global_time']
                if interval < full_step:
                    full_step = interval

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
            if not isinstance(series, list):
                series = [series]

            for update in series:
                paths = hierarchy_depth(update)
                update_paths.extend(paths.keys())

                self.state = types.apply_update(
                    self.composition,
                    self.state,
                    update)

                bridge_update = types.view(
                    self.process_schema,
                    self.bridge,
                    (),
                    update)

                if bridge_update:
                    self.bridge_updates.append(bridge_update)

        return update_paths

                # view_expire_update = self.apply_update(up, store)
                # view_expire = view_expire or view_expire_update

        # if view_expire:
        #     self.state.build_topology_views()


    def run(self, interval, force_complete=False):
        end_time = self.state['global_time'] + interval
        while self.state['global_time'] < end_time or force_complete:
            full_step = math.inf

            for path, process in self.process_paths.items():
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

                state = types.view_edge(
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


    def gather_results(self, queries=None):
        '''
        a map of paths to emitters --> queries for the emitter at that path
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

        projection = types.project(
            self.schema(),
            self.bridge,
            [],
            state)

        self.state = types.set(
            self.composition,
            self.state,
            projection)

        self.run(interval)

        updates = self.bridge_updates
        self.bridge_updates = []

        return updates


