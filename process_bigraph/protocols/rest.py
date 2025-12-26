"""
===============================================
Protocol for running processes in parallel using
python multiprocessing
===============================================
"""

import sys
import json
import time
import uuid
import pstats
import socket

from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from urllib.parse import urlparse, urlunparse
import requests

from bigraph_schema.schema import Node, String, Protocol
from bigraph_schema.methods import load_protocol

from process_bigraph.composite import Process, SyncUpdate


def rest_get(url, parameters=None):
    return requests.get(
        urlunparse(url),
        json=parameters).json()


def rest_post(url, parameters=None):
    return requests.post(
        urlunparse(url),
        json=parameters).json()


@dataclass(kw_only=True)
class RestData(Node):
    process: String = field(default_factory=String)
    host: String = field(default_factory=String)
    port: String = field(default_factory=String)


@dataclass(kw_only=True)
class RestProtocol(Protocol):
    data: RestData = field(default_factory=RestData)


class RestProcess(Process):
    def __init__(self, data, config, core) -> None:
        self._ended = False

        self.process_name = data['process']

        self.base_url = data['base_url'] or urlparse('http://localhost:22222')

        self.initialize_url = self.base_url._replace(
            path=f'/process/{self.process_name}/initialize')
        self.process_id = rest_post(
            self.initialize_url,
            config)

        self.end_url = self.base_url._replace(
            path=f'/process/{self.process_name}/end/{self.process_id}')
        self.inputs_url = self.base_url._replace(
            path=f'/process/{self.process_name}/inputs/{self.process_id}')
        self.outputs_url = self.base_url._replace(
            path=f'/process/{self.process_name}/outputs/{self.process_id}')
        self.update_url = self.base_url._replace(
            path=f'/process/{self.process_name}/update/{self.process_id}')

        super().__init__(config, core=core)

    def get(self):
        return self.get_command_result()

    @staticmethod
    def generate_state(config=None):
        """Generate static initial state for user configuration or inspection."""
        return {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            '_type': 'process',
            '_inputs': self.inputs(),
            '_outputs': self.outputs()}

    @schema.setter
    def schema(self, schema):
        pass

    @property
    def state(self) -> Dict[str, Any]:
        return self

    @state.setter
    def state(self, state):
        pass

    def initial_state(self):
        """Return initial state values, if applicable."""
        return {}

    def inputs(self):
        """
        Return a dictionary mapping input port names to bigraph types.

        Example:
            {'glucose': 'float', 'biomass': 'map[float]'}
        """
        response = rest_get(self.inputs_url)

        return response

    def outputs(self):
        """
        Return a dictionary mapping output port names to bigraph types.

        Example:
            {'growth_rate': 'float'}
        """
        response = rest_get(self.outputs_url)

        return response

    def update(self, state, interval):
        response = rest_post(self.update_url, {
            'state': state,
            'interval': interval})

        return response

    def end(self) -> None:
        """
        remove the container
        """
        # Only end once.
        if self._ended:
            return

        rest_post(self.end_url)

        self._ended = True

    def __del__(self) -> None:
        self.end()


@load_protocol.dispatch
def load_protocol(core, protocol: RestProtocol, data):
    ssh = ''
    process_name = data['process']
    host = data['host']
    port = data['port']
    base_raw = f'http{ssh}://{host}:{port}'
    base_url = urlparse(base_raw)

    config_schema_url = base_url._replace(
        path=f'/process/{process_name}/config-schema')
    config_schema = rest_get(
        config_schema_url)

    instance = {
        'base_url': base_url,
        'process': process_name}

    def instantiate(config, core=None):
        return RestProcess(
            instance,
            config,
            core)

    instantiate.config_schema = config_schema
    return instantiate


def register_types(core):
    core.register_types({
        'rest': RestProtocol})

    return core

