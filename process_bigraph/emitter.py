import copy
import os
import json
import uuid
import orjson
import itertools
from functools import partial
from warnings import warn
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from urllib.parse import quote_plus
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymongo import ASCENDING
from pymongo.errors import DocumentTooLarge
from pymongo.mongo_client import MongoClient
from bson import MinKey, MaxKey

from bigraph_schema import get_path, set_path
from process_bigraph.composite import Step, Process
from process_bigraph import process_registry


HISTORY_INDEXES = [
    'data.time',
    [('experiment_id', ASCENDING),
     ('data.time', ASCENDING),
     ('_id', ASCENDING)],
]

CONFIGURATION_INDEXES = [
    'experiment_id',
]

SECRETS_PATH = 'secrets.json'


class Emitter(Step):
    def query(self, query=None):
        return {}


class ConsoleEmitter(Emitter):
    config_schema = {
        'ports': 'tree[any]'}


    def schema(self):
        return self.config['ports']


    def update(self, state):
        print(state)

        return {}


class RAMEmitter(Emitter):
    config_schema = {
        'ports': 'tree[any]'}


    def __init__(self, config):
        super().__init__(config)

        self.history = []


    def schema(self):
        return self.config['ports']


    def update(self, state):
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


class DatabaseEmitter(Emitter):
    """
    Emit data to a mongoDB database

    Example:

    >>> config = {
    ...     'host': 'localhost:27017',
    ...     'database': 'DB_NAME',
    ... }
    >>> # The line below works only if you have to have 27017 open locally
    >>> # emitter = DatabaseEmitter(config)
    """
    default_host = 'localhost:27017'
    client_dict: Dict[int, MongoClient] = {}

    @classmethod
    def create_indexes(cls, table: Any, columns: List[Any]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config: Dict[str, Any]) -> None:
        """config may have 'host' and 'database' items."""
        super().__init__(config)
        self.experiment_id = config.get('experiment_id')
        # In the worst case, `breakdown_data` can underestimate the size of
        # data by a factor of 4: len(str(0)) == 1 but 0 is a 4-byte int.
        # Use 4 MB as the breakdown limit to stay under MongoDB's 16 MB limit.
        self.emit_limit = config.get('emit_limit', 4000000)
        self.embed_path = config.get('embed_path', tuple())

        # create new MongoClient per OS process
        curr_pid = os.getpid()
        if curr_pid not in DatabaseEmitter.client_dict:
            DatabaseEmitter.client_dict[curr_pid] = MongoClient(
                config.get('host', self.default_host))
        self.client = DatabaseEmitter.client_dict[curr_pid]

        self.db = getattr(self.client, config.get('database', 'simulations'))
        self.history = getattr(self.db, 'history')
        self.configuration = getattr(self.db, 'configuration')
        self.phylogeny = getattr(self.db, 'phylogeny')
        self.create_indexes(self.history, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)
        self.create_indexes(self.phylogeny, CONFIGURATION_INDEXES)

        self.fallback_serializer = make_fallback_serializer_function()

    def emit(self, data: Dict[str, Any]) -> None:
        table_id = data['table']
        table = self.db.get_collection(table_id)
        time = data['data'].pop('time', None)
        data['data'] = assoc_path({}, self.embed_path, data['data'])
        # Analysis scripts expect the time to be at the top level of the
        # dictionary, but some emits, like configuration emits, lack a
        # time key.
        if time is not None:
            data['data']['time'] = time
        emit_data = data.copy()
        emit_data.pop('table', None)
        emit_data['experiment_id'] = self.experiment_id
        self.write_emit(table, emit_data)

    def write_emit(self, table: Any, emit_data: Dict[str, Any]) -> None:
        """Check that data size is less than emit limit.

        Break up large emits into smaller pieces and emit them individually
        """
        assembly_id = str(uuid.uuid4())
        emit_data = serialize_value(emit_data, self.fallback_serializer)
        try:
            emit_data['assembly_id'] = assembly_id
            table.insert_one(emit_data)
        # If document is too large, break up into smaller dictionaries
        # with shared assembly IDs and time keys
        except DocumentTooLarge:
            emit_data.pop('assembly_id')
            # insert_one adds this key to emit_data
            emit_data.pop('_id')
            experiment_id = emit_data.pop('experiment_id')
            time = emit_data['data'].pop('time', None)
            broken_down_data = breakdown_data(self.emit_limit, emit_data)
            for (path, datum) in broken_down_data:
                d: Dict[str, Any] = {}
                assoc_path(d, path, datum)
                d['assembly_id'] = assembly_id
                d['experiment_id'] = experiment_id
                if time:
                    d.setdefault('data', {})
                    d['data']['time'] = time
                table.insert_one(d)

    def get_data(self, query: Optional[list] = None) -> dict:
        return get_history_data_db(self.history, self.experiment_id, query)


def make_fallback_serializer_function() -> Callable:
    """Creates a fallback function that is called by orjson on data of
    types that are not natively supported. Define and register instances of
    :py:class:`vivarium.core.registry.Serializer()` with serialization
    routines for the types in question."""

    def default(obj: Any) -> Any:
        # Try to lookup by exclusive type
        serializer = process_registry.access(str(type(obj)))
        if not serializer:
            compatible_serializers = []
            for serializer_name in process_registry.list():
                test_serializer = process_registry.access(serializer_name)
                # Subclasses with registered serializers will be caught here
                if isinstance(obj, test_serializer.python_type):
                    compatible_serializers.append(test_serializer)
            if len(compatible_serializers) > 1:
                raise TypeError(
                    f'Multiple serializers ({compatible_serializers}) found '
                    f'for {obj} of type {type(obj)}')
            if not compatible_serializers:
                raise TypeError(
                    f'No serializer found for {obj} of type {type(obj)}')
            serializer = compatible_serializers[0]
            if not isinstance(obj, Process):
                # We don't warn for processes because since their types
                # based on their subclasses, it's not possible to avoid
                # searching through the serializers.
                warn(
                    f'Searched through serializers to find {serializer} '
                    f'for data of type {type(obj)}. This is '
                    f'inefficient.')
        return serializer.serialize(obj)
    return default


def find_numpy_and_non_strings(
    d: dict,
    curr_path: Tuple = tuple(),
    saved_paths: Optional[List[Tuple]] = None
) -> List[Tuple]:
    """Return list of paths which terminate in a non-string or Numpy string
    dictionary key. Orjson does not handle these types of keys by default."""
    if not saved_paths:
        saved_paths = []
    if isinstance(d, dict):
        for key in d.keys():
            if not isinstance(key, str):
                saved_paths.append(curr_path + (key,))
            elif isinstance(key, np.str_):
                saved_paths.append(curr_path + (key,))
            saved_paths = find_numpy_and_non_strings(
                d[key], curr_path+(key,), saved_paths)
    return saved_paths


def serialize_value(
    value: Any,
    default: Optional[Callable] = None,
) -> Any:
    """Apply orjson-based serialization routine on ``value``.

    Args:
        value (Any): Data to be serialized. All keys must be strings. Notably,
            Numpy strings (``np.str_``) are not acceptable keys.
        default (Callable): A function that is called on any data of a type
            that is not natively supported by orjson. Returns an object that
            can be handled by default up to 254 times before an exception is
            raised.

    Returns:
        Any: Serialized data
    """
    if default is None:
        default = make_fallback_serializer_function()
    try:
        value = orjson.dumps(
            value, option=orjson.OPT_SERIALIZE_NUMPY,
            default=default
        )
        return orjson.loads(value)
    except TypeError as e:
        bad_keys = find_numpy_and_non_strings(value)
        raise TypeError('These paths end in incompatible non-string or Numpy '
            f'string keys: {bad_keys}').with_traceback(e.__traceback__) from e
