"""
Composite and Process classes
"""

import abc
import copy
import math

from bigraph_schema.registry import deep_merge
from process_bigraph.type_system import types, lookup_local


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


class Step():
    # TODO: support trigger every time
    #   as well as dependency trigger
    config_schema = {}


    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = types.fill(
            self.config_schema,
            config)
        

    def schema(self):
        return {}


    def invoke(self, state):
        update = self.update(state)
        sync = SyncUpdate(update)

        return sync

    @abc.abstractmethod
    def update(self, state):
        return {}


class Process():
    config_schema = {}

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = types.fill(
            self.config_schema,
            config)


    @abc.abstractmethod
    def schema(self):
        return {}


    def initial_state(self, initial=None):
        initial = initial or {}
        return types.fill(
            self.schema(),
            initial)
        

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
    def __init__(
            self,
            defer,
            f,
            args):
            # defer: Any,
            # f: Callable,
            # args: Tuple,
    # ) -> None:
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

# maybe keep wires as tuples/paths to distinguish them from schemas?


def find_instances(state, instance_type='process_bigraph.composite.Process'):
    process_class = lookup_local(instance_type)
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
    prefix = path[:-1]
    triggers = {}
    for wire in step['wires']['inputs'].values():
        trigger_path = tuple(prefix) + tuple(wire)
        triggers[trigger_path] = path

    return triggers


def empty_front(time):
    return {
        'time': time,
        'update': {}
    }


class Composite(Process):
    config_schema = {
        # TODO: add schema type
        'composition': 'tree[any]', # 'schema',
        'state': 'tree[any]',
        'schema': 'tree[any]', # 'schema',
        'bridge': 'wires',
        'initial_time': 'float',
        'global_time_precision': 'maybe[float]',
    }

    # TODO: if processes are serialized, deserialize them first
    def __init__(self, config=None):
        super().__init__(config)

        self.composition = types.access(self.config['composition'])
        self.composition = copy.deepcopy(self.composition)

        self.state = types.hydrate(
            self.composition,
            self.config['state'])

        self.global_time = self.config['initial_time']
        self.global_time_precision = self.config['global_time_precision']

        self.process_paths = find_instance_paths(
            self.state,
            'process_bigraph.composite.Process')

        self.step_paths = find_instance_paths(
            self.state,
            'process_bigraph.composite.Step')

        import ipdb; ipdb.set_trace()

        self.step_triggers = {}

        for step_path, step in self.step_paths.items():
            step_triggers = find_step_triggers(
                step_path, step)
            # TODO: merge all triggers together

        self.front: Dict = {
            path: empty_front(self.global_time)
            for path in self.process_paths}

    def schema(self):
        return self.config['schema']

    def process_update(
            self,
            path,
            process,
            states,
            interval,
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
            store: The store at ``path``.
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
                update)

        absolute = Defer(
            update,
            defer_project, (
                self.config['composition'],
                self.state,
                path))

        return absolute

    def run_process(self, path, process, end_time, full_step, force_complete):
        if path not in self.front:
            self.front[path] = empty_front(self.global_time)
        process_time = self.front[path]['time']
        if process_time <= self.global_time:
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
                interval = future - self.global_time
                if interval < full_step:
                    full_step = interval
            else:
                # absolute interval
                interval = future - self.global_time
                if interval < full_step:
                    full_step = interval

        else:
            # don't shoot past processes that didn't run this time
            process_delay = process_time - self.global_time
            if process_delay < full_step:
                full_step = process_delay

        return full_step

    def apply_updates(self, updates):
        # view_expire = False
        resolved = []
        for defer in updates:
            series = defer.get()
            if not isinstance(series, list):
                series = [series]

            resolved.append(series)

            for update in series:
                self.state = types.apply(
                    self.composition,
                    self.state,
                    update)

        # TODO: keep track of updated paths so we can trigger steps

                # view_expire_update = self.apply_update(up, store)
                # view_expire = view_expire or view_expire_update

        # if view_expire:
        #     self.state.build_topology_views()

        # self.run_steps()

    def run(self, interval, force_complete=False):
        end_time = self.global_time + interval
        while self.global_time < end_time or force_complete:
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
                self.global_time = next_event

            elif self.global_time + full_step <= end_time:
                # at least one process ran within the interval
                # increase the time, apply updates, and continue
                self.global_time += full_step

                # apply updates that are behind global time
                updates = []
                paths = []
                for path, advance in self.front.items():
                    if advance['time'] <= self.global_time \
                            and advance['update']:
                        new_update = advance['update']
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                self.apply_updates(updates)

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
                self.global_time = end_time

            if force_complete and self.global_time == end_time:
                force_complete = False
        

    def update(self, state, interval):
        # do everything

        # this needs to go through the bridge
        projection = types.project(
            self.schema(),
            self.config['bridge'],
            [],
            state)

        # TODO: this may need to be a set instead of an update
        #   add a force set?
        self.state = types.apply(
            self.composition,
            self.state,
            projection)

        self.run(interval)

        # pull the update out of the state and return it
        # TODO: this is the state, but we need to return an update
        #   store all updates to the bridge internally, then return them
        #   as the update
        update = types.view(
            self.schema(),
            self.config['bridge'],
            [],
            self.state)

        return update


class Generator():
    def __init__(self, config):
        self.config = config

    def __call__(self, config=None):
        config = deep_merge(self.config, config)
        return Composite(config)


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'}}

    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'level': 'float'}

    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']}


def test_default_config():
    process = IncreaseProcess()
    assert process.config['rate'] == 0.1


def test_process():
    process = IncreaseProcess({'rate': 0.2})
    schema = process.schema()
    state = types.fill(schema)
    update = process.update({'level': 5.5}, 1.0)
    new_state = types.apply(schema, state, update)

    assert new_state['level'] == 1.1


def test_composite():
    # TODO: add support for the various vivarium emitters

    # increase = IncreaseProcess({'rate': 0.3})
    # TODO: This is the config of the composite,
    #   we also need a way to serialize the entire composite

    composite = Composite({
        'composition': {
            'increase': 'process[level:float]',
            'value': 'float'},
        'schema': {
            'exchange': 'float'},
        'bridge': {
            'exchange': ['value']},
        'state': {
            # TODO: interval is state?
            'increase': {
                'address': 'local:process_bigraph.composite.IncreaseProcess',
                'config': {'rate': '0.3'},
                'wires': {'level': ['value']}},
            'value': '11.11'}})

    composite.update({'exchange': 3.33}, 10.0)

    assert composite.state['value'] > 199


def test_serialized_composite():
    # This should specify the same thing as above
    composite_schema = {
        '_type': 'process[exchange:float]',
        'address': 'local:Composite',
        'config': {
            'state': {
                'increase': {
                    '_type': 'process[level:float]',
                    'address': 'local:IncreaseProcess',
                    'config': {'rate': '0.3'},
                    'wires': {'level': ['value']}
                },
                'value': '11.11',
            },
            'schema': {
                'increase': 'process[level:float]',
                # 'increase': 'process[{"level":"float","down":{"a":"int"}}]',
                'value': 'float',
            },
            'bridge': {
                'exchange': 'value'
            },
        }
    }

    composite_instance = types.deserialize(composite_schema)
    composite_instance.update()


if __name__ == '__main__':
    test_default_config()
    test_process()
    test_composite()
