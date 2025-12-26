import random

from bigraph_schema import make_default
from process_bigraph.composite import Process, Step, Composite


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'}}

    def inputs(self):
        return {
            'level': 'float'}

    def outputs(self):
        return {
            'level': 'float'}

    def accelerate(self, delta):
        self.config['rate'] += delta

    def initial_state(self):
        return {
            'level': 4.4}

    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']}


class IncreaseRate(Step):
    config_schema = {
        'acceleration': make_default('float', 0.001)}

    def inputs(self):
        return {
            'level': 'float'}

    def update(self, state):
        # TODO: this is ludicrous.... never do this
        #   probably this feature should only be used for reading
        self.instance.accelerate(
            self.config['acceleration'] * state['level'])


class OperatorStep(Step):
    config_schema = {
        'operator': 'string'}


    def inputs(self):
        return {
            'a': 'float',
            'b': 'float'}


    def outputs(self):
        return {
            'c': 'float'}


    def update(self, inputs):
        a = inputs['a']
        b = inputs['b']

        if self.config['operator'] == '+':
            c = a + b
        elif self.config['operator'] == '*':
            c = a * b
        elif self.config['operator'] == '-':
            c = a - b

        return {'c': c}


class SimpleCompartment(Process):
    config_schema = {
        'id': 'string'}


    def interface(self):
        return {
            'outer': 'tree[process]',
            'inner': 'tree[process]'}


    def update(self, state, interval):
        choice = random.random()
        update = {}

        outer = state['outer']
        inner = state['inner']

        # TODO: implement divide_state(_)
        divisions = self.core.divide_state(
            self.interface(),
            inner)

        if choice < 0.2:
            # update = {
            #     'outer': {
            #         '_divide': {
            #             'mother': self.config['id'],
            #             'daughters': [
            #                 {'id': self.config['id'] + '0'},
            #                 {'id': self.config['id'] + '1'}]}}}

            # daughter_ids = [self.config['id'] + str(i)
            #     for i in range(2)]

            # update = {
            #     'outer': {
            #         '_react': {
            #             'redex': {
            #                 'inner': {
            #                     self.config['id']: {}}},
            #             'reactum': {
            #                 'inner': {
            #                     daughter_config['id']: {
            #                         '_type': 'process',
            #                         'address': 'local:SimpleCompartment',
            #                         'config': daughter_config,
            #                         'inner': daughter_inner,
            #                         'wires': {
            #                             'outer': ['..']}}
            #                     for daughter_config, daughter_inner in zip(daughter_configs, divisions)}}}}}

            update = {
                'outer': {
                    'inner': {
                        '_react': {
                            'reaction': 'divide',
                            'config': {
                                'id': self.config['id'],
                                'daughters': [{
                                        'id': daughter_id,
                                        'state': daughter_state}
                                    for daughter_id, daughter_state in zip(
                                        daughter_ids,
                                        divisions)]}}}}}

        return update


class WriteCounts(Step):
    def inputs(self):
        return {
            'volumes': 'map[float]',
            'concentrations': 'map[map[float]]'}


    def outputs(self):
        return {
            'counts': 'map[map[overwrite[integer]]]'}


    def update(self, state):
        counts = {}

        for key, local in state['concentrations'].items():
            counts[key] = {}
            for substrate, concentration in local.items():
                count = int(concentration * state['volumes'][key])
                counts[key][substrate] = count

        return {
            'counts': counts}


class AboveProcess(Process):
    config_schema = {
        'rate': 'float'}

    def inputs(self):
        return {
            'below': 'map[mass:float]'}

    def outputs(self):
        return {
            'below': 'map[mass:float]'}

    def update(self, state, interval):
        update = {
            'below': {}}

        for id, pod in state['below'].items():
            update['below'][id] = {
                'mass': self.config['rate'] * pod['mass']}

        return update


class BelowProcess(Process):
    next_id = 1

    config_schema = {
        'id': 'string',
        'creation_probability': 'float',
        'annihilation_probability': 'float'}

    def inputs(self):
        return {
            'mass': 'float',
            'entropy': 'float'}

    def outputs(self):
        return {
            'entropy': 'float',
            'environment': 'map[mass:float]'}

    def update(self, state, interval):
        creation = random.random() < self.config['creation_probability'] * state['entropy']
        annihilation = random.random() < self.config['annihilation_probability'] * state['entropy']

        update = {}

        if creation or annihilation:
            update['environment'] = {}

        if creation:
            new_id = str(BelowProcess.next_id)
            BelowProcess.next_id += 1

            update['environment']['_add'] = {
                new_id: {
                    'mass': 1.1,
                    'below': {
                        'config': {
                            'id': new_id}}}}

        if annihilation:
            update['environment']['_remove'] = [self.config['id']]

        if not 'environment' in update:
            update['entropy'] = 0.1
        else:
            update['entropy'] = -state['entropy']

        return update


