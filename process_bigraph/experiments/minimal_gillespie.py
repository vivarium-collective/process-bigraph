""" Toy Stochastic Transcription Process
Toy model of Gillespie algorithm-based  transcription,
and a composite with deterministic translation.

Note: This Process is primarily for testing multi-timestepping.
variables and parameters are hard-coded. Do not use this as a
general stochastic transcription.
"""


import numpy as np
import pytest

from process_bigraph.composite import Step, Process, Composite, ProcessEnsemble


class GillespieInterval(Step):
    config_schema = {
        'ktsc': {
            '_type': 'float',
            '_default': '5e0'},
        'kdeg': {
            '_type': 'float',
            '_default': '1e-1'}}


    def inputs(self):
        return {
            'DNA': 'map[default 1]',
            'mRNA': {
                'A mRNA': 'default 1',
                'B mRNA': 'default 1'}}

                    # {
                    # '_type': 'map',
                    # '_value': 'float(default:1.0)'},

                    # 'G': {
                    #     '_type': 'float',
                    #     '_default': '1.0'}},


    def outputs(self):
        return {
            'interval': 'interval'}


    def initial_state(self):
        return {
            'mRNA': {
                'A mRNA': 2.0,
                'B mRNA': 3.0}}


    def update(self, input):
        # retrieve the state values
        g = input['DNA']['A gene']
        c = input['mRNA']['A mRNA']

        array_state = np.array([g, c])

        # Calculate propensities
        propensities = [
            self.config['ktsc'] * array_state[0],
            self.config['kdeg'] * array_state[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        interval = np.random.exponential(scale=prop_sum)

        output = {
            'interval': interval}

        # print(f'produced interval: {output}')

        return output


class GillespieEvent(Process):
    """stochastic toy transcription"""
    config_schema = {
        'ktsc': {
            '_type': 'float',
            '_default': '5e0'},
        'kdeg': {
            '_type': 'float',
            '_default': '1e-1'}}


    def initialize(self, config=None):
        self.stoichiometry = np.array([[0, 1], [0, -1]])


    def initial_state(self):
        return {
            'mRNA': {
                'C mRNA': 11.111},
            'DNA': {
                'A gene': 3.0,
                'B gene': 5.0}}


    def inputs(self):
        return {
            'mRNA': 'map[float]',
            'DNA': {
                'A gene': 'float',
                'B gene': 'float'}}

    def outputs(self):
        return {
            'mRNA': 'map[float]'}


    def next_reaction(self, x):
        """get the next reaction and return a new state"""

        propensities = [
            self.config['ktsc'] * x[0],
            self.config['kdeg'] * x[1]]
        prop_sum = sum(propensities)

        # Choose the next reaction
        r_rxn = np.random.uniform()
        i = 0
        for i, _ in enumerate(propensities):
            if r_rxn < propensities[i] / prop_sum:
                # This means propensity i fires
                break
        x += self.stoichiometry[i]
        return x


    def update(self, state, interval):

        # retrieve the state values, put them in array
        g = state['DNA']['A gene']
        c = state['mRNA']['A mRNA']
        array_state = np.array([g, c])

        # calculate the next reaction
        new_state = self.next_reaction(array_state)

        # get delta mRNA
        c1 = new_state[1]
        d_c = c1 - c

        update = {
            'mRNA': {
                'A mRNA': d_c}}

        # print(f'received interval: {interval}')

        return update


class GillespieSimulation(ProcessEnsemble):
    def __init__(self, config=None, core=None):
        super.__init__(config, core)


    def inputs_interval(self):
        return {
            'DNA': 'map[default 1]',
            'mRNA': {
                'A mRNA': 'default 1',
                'B mRNA': 'default 1'}}

                    # {
                    # '_type': 'map',
                    # '_value': 'float(default:1.0)'},

                    # 'G': {
                    #     '_type': 'float',
                    #     '_default': '1.0'}},


    def outputs_interval(self):
        return {
            'interval': 'interval'}


    # def interface_interval(self):


    def calculate_interval(self, inputs):
        # retrieve the state values
        g = input['DNA']['A gene']
        c = input['mRNA']['A mRNA']

        array_state = np.array([g, c])

        # Calculate propensities
        propensities = [
            self.config['ktsc'] * array_state[0],
            self.config['kdeg'] * array_state[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        interval = np.random.exponential(scale=prop_sum)

        output = {
            'interval': interval}

        print(f'produced interval: {output}')

        return output
