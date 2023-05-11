import abc
from bigraph_schema import fill, registry_registry, type_registry, apply_update


def hierarchy_depth(hierarchy, path=()):
    """
    Create a mapping of every path in the hierarchy to the node living at
    that path in the hierarchy.
    """

    base = {}

    for key, inner in hierarchy.items():
        down = tuple(path + (key,))
        if isinstance(inner, dict):
            base.update(hierarchy_depth(inner, down))
        else:
            base[down] = inner

    return base


def find_processes(state):
    found = {}

    for key, inner in state.items():
        if isinstance(inner, Process):
            found[key] = inner
        elif isinstance(inner, dict):
            result = find_processes(inner)
            if result:
                found[key] = result

    return found


def find_process_paths(state):
    processes = find_processes(state)
    return hierarchy_depth(processes)


def empty_front(time):
    return {
        'time': time,
        'update': {}
    }

# deal with steps vs temporal process vs edges

class Process:
    config_schema = {}

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config_schema.setdefault(
            'timestep',
            {'_type': 'float', '_default': '1.0'}
        )

        self.config = fill(
            self.config_schema,
            config
        )

    @abc.abstractmethod
    def schema(self):
        return {}

    def calculate_timestep(self, state):
        return self.config['timestep']

    @abc.abstractmethod
    def update(self, state, interval):
        return {}

    # TODO: should we include run(interval) here?
    #   process would have to maintain state


class Composite(Process):
    config_schema = {
        'schema': 'tree[any]',
        'bridge': 'wires',
        'instance': 'tree[any]',
        'initial_time': 'float',
        'global_time_precision': 'maybe[float]',
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.global_time = self.config['initial_time']
        self.state = fill(
            self.config['schema'],
            self.config['instance']
        )
        self.process_paths = find_process_paths(state)
        self.front: Dict = {
            path: empty_front(self.global_time)
            for path in self.process_paths
        }
        self.global_time_precision = self.config['global_time_precision']

    def schema(self):
        return self.config['schema']

    def run_process(self, path, process):
        if path not in self.front:
            self.front[path] = empty_front(self.global_time)
        process_time = self.front[path]['time']
        if process_time <= self.global_time:
            if self.front[path].get('future'):
                future_front = self.front[path]['future']
                process_timestep = future_front['timestep']
                store = future_front['store']
                state = future_front['state']
                del self.front[path]['future']
            else:
                # get the time step
                store, state = self._process_state(path)
                process_timestep = process.calculate_timestep(state)

            if force_complete:
                # force the process to complete at end_time
                future = min(process_time + process_timestep, end_time)
            else:
                future = process_time + process_timestep
            if self.global_time_precision is not None:
                # set future time based on global_time_precision
                future = round(future, self.global_time_precision)

            if future <= end_time:

                update = self._process_update(
                    path,
                    process,
                    store,
                    state,
                    process_timestep
                )

                # update front, to be applied at its projected time
                self.front[path]['time'] = future
                self.front[path]['update'] = update

                # absolute timestep
                timestep = future - self.global_time
                if timestep < full_step:
                    full_step = timestep
            else:
                # absolute timestep
                timestep = future - self.global_time
                if timestep < full_step:
                    full_step = timestep

        else:
            # don't shoot past processes that didn't run this time
            process_delay = process_time - self.global_time
            if process_delay < full_step:
                full_step = process_delay

        return full_step
        
    def run(self, interval, force_complete=False):
        end_time = self.global_time + interval
        while self.global_time < end_time or force_complete:
            full_step = math.inf
            for path, process in self.process_paths.items():
                full_step = self.run_process(process, path)

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

                # advance all quiet processes to current time
                for quiet in quiet_paths:
                    self.front[quiet]['time'] = self.global_time

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

                self._send_updates(updates)

                # display and emit
                if self.progress_bar:
                    print_progress_bar(self.global_time, end_time)
                if self.emit_step == 1:
                    self._emit_store_data()
                elif emit_time <= self.global_time:
                    while emit_time <= self.global_time:
                        self._emit_store_data()
                        emit_time += self.emit_step

            else:
                # all processes have run past the interval
                self.global_time = end_time

            if force_complete and self.global_time == end_time:
                force_complete = False

    def update(self, state, interval):
        # do everything

        # this needs to go through the bridge
        self.state = apply_update(
            self.schema,
            self.state,
            state
        )

        self.run(interval)

        # pull the update out of the state and return it


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1',
        }
    }

    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'level': 'float',
        }
    
    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']
        }


def test_process():
    process = IncreaseProcess({'rate': 0.2})
    schema = process.schema()
    state = fill(schema)
    update = process.update({'level': 5.5}, 1.0)
    new_state = apply_update(schema, state, update)
    assert new_state['level'] == 1.1


{"level":"float","down":{"a":"int"}}

def test_composite():
    # TODO: add support for the various vivarium emitters

    increase = IncreaseProcess({'rate': 0.3})
    composite = Composite({
        'schema': {
            'increase': 'process[level:float]',
            # 'increase': 'process[{"level":"float","down":{"a":"int"}}]',
            'value': 'float',
        },
        'bridge': {
            'exchange': 'value'
        },
        'instance': {
            'increase': increase,
            'wires': {
                'level': 'value'
            },
            'value': 11.11,
        },
    })

    composite.update({'exchange': 3.33}, 10.0)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_process()
    test_composite()
