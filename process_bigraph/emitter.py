import copy
import os
import uuid
import orjson
import collections
import itertools
from functools import partial
from warnings import warn
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
# from types import NoneType
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pymongo import ASCENDING
from pymongo.mongo_client import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DocumentTooLarge
from pymongo.database import Database
from bigraph_schema import get_path, set_path
from bson import MinKey, MaxKey
from process_bigraph.composite import Step, Process
from process_bigraph.registry import process_registry


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

    def __init__(self, config):
        super().__init__(config)
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


class DatabaseEmitter(Emitter):
    """
        Emit data to a mongoDB database

        Attributes:
            client_dict:`Dict[str, MongoClient]`

        >>> # The line below works only if you have to have 27017 open locally
        >>> # emitter = DatabaseEmitter(config)

        PLEASE NOTE: For Mac Silicon, you can start a Mongo instance as a background process with:

            ``mongod --config /opt/homebrew/etc/mongod.conf --fork``

            ...and the process can be stopped in two ways:

                1.) `kill {PID}`

                2.) `mongosh -> use admin -> db.shutdownServer()`
    """
    client_dict: Dict[int, MongoClient] = {}
    config_schema = {
        'experiment_id': {
            '_type': 'string',
            '_default': str(uuid.uuid4())
        },
        'emit_limit': {
            '_type': 'int',
            '_default': 4000000
        },
        'embed_path': {
            '_type': 'list[string]',
        },
        'host': {
            '_type': 'string',
            '_default': 'localhost:27017'
        },
        'database': {
            '_type': 'string',
            '_default': 'simulations'
        },
    }


    @classmethod
    def create_indexes(cls, table: Any, columns: List[Any]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """Config may have 'host' and 'database' items. The config passed is expected to be:

                {'experiment_id':,
                 'emit_limit':,
                 'embed_path':}

                TODO: Automate this process for the user in builder
        """
        super().__init__(config)
        self.experiment_id = self.config['experiment_id']
        # In the worst case, `breakdown_data` can underestimate the size of
        # data by a factor of 4: len(str(0)) == 1 but 0 is a 4-byte int.
        # Use 4 MB as the breakdown limit to stay under MongoDB's 16 MB limit.
        self.emit_limit = self.config['emit_limit']
        self.embed_path = self.config['embed_path']

        # create new MongoClient per OS process
        curr_pid = os.getpid()
        if curr_pid not in DatabaseEmitter.client_dict:
            DatabaseEmitter.client_dict[curr_pid] = MongoClient(
                config['host'])
        self.client: MongoClient = DatabaseEmitter.client_dict[curr_pid]

        # extract objects from current mongo client instance
        self.db: Database = getattr(self.client, self.config.get('database', 'simulations'))
        self.history: Collection = getattr(self.db, 'history')
        self.configuration: Collection = getattr(self.db, 'configuration')
        self.phylogeny: Collection = getattr(self.db, 'phylogeny')

        # create column indexes for the given collection objects
        self.create_indexes(self.history, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)
        self.create_indexes(self.phylogeny, CONFIGURATION_INDEXES)

        self.fallback_serializer = make_fallback_serializer_function()

    def write_emit(self, table: Collection, emit_data: Dict[str, Any]) -> None:
        """Check that data size is less than emit limit. Break up large emits into smaller pieces and
            emit them individually.

            Args:
                table:`pymongo.collection.Collection`: pymongo collection to which data will be written.
                emit_data:`Dict[str, Any]`: Data to be passed and saved to the collection.
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

    def query(self, query: Optional[List[Tuple[str]]] = None) -> Dict:
        """API contract-wrapper for `self.get_data`. Get data based on the passed query.

             Args:
                 query: a list of tuples pointing to fields within the experiment data.
                     In the format: [('path', 'to', 'field1'), ('path', 'to', 'field2')]

            Returns:
                 `Dict`
        """
        return get_history_data_db(self.history, self.experiment_id, query)

    def update(self, state):
        if not state.get('table'):
            state['table'] += self.config['database']
        table_id = state['table']
        table = self.db.get_collection(table_id)
        time = state['data'].pop('time', None)
        state['data'] = assoc_path({}, self.embed_path, state['data'])
        # Analysis scripts expect the time to be at the top level of the
        # dictionary, but some emits, like configuration emits, lack a
        # time key.
        if time is not None:
            state['data']['time'] = time
        emit_data = state.copy()
        emit_data.pop('table', None)
        emit_data['experiment_id'] = self.experiment_id
        self.write_emit(table, emit_data)
        print(emit_data)
        return {}



def format_data(table_id: str, time: Optional[Union[int, str]] = None, **values: Any) -> Dict[str, Any]:
    """Format the given data for mongo db emission.
        Args:
            table_id:`str`: id of the table of insertion. Usually, this value is some sort of simulation run id.
            time:`Optional[Union[int, str]]`: Timestamp by which the table will be indexed and data retrieved.
                Defaults to `None`.
            **values: Data values to insert into the db. Kwargs will be related only to the data being stored.
        Returns:
            `Dict`: formatted data with the typeshape: `{str: Union[str, Tuple, List, Dict, np.ndarray]]}`
    """
    return {
        'table': table_id,
        'data': {
            'time': time,
            'values': {**values}
        }
    }


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


def assoc_path(d: Dict, path, value):
    '''Insert ``value`` into the dictionary ``d`` at ``path``.

    >>> d = {'a': {'b': 'c'}}
    >>> assoc_path(d, ('a', 'd'), 'e')
    {'a': {'b': 'c', 'd': 'e'}}
    >>> d
    {'a': {'b': 'c', 'd': 'e'}}

    Create new dictionaries recursively as needed.
    '''

    if path:
        head = path[0]
        if len(path) == 1:
            d[head] = value
        else:
            if head not in d:
                d[head] = {}
            assoc_path(d[head], path[1:], value)
    elif isinstance(value, dict):
        deep_merge(d, value)
    return d


def deep_merge(dct: Optional[Dict] = None, merge_dct: Optional[Dict] = None) -> Dict:
    """ Recursive dict merge

    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def breakdown_data(
        limit: float,
        data: Any,
        path: Tuple = (),
        size: Optional[float] = None,
        ) -> List[Tuple]:
    size = size or len(str(data))
    if size > limit:
        if isinstance(data, dict):
            output = []
            subsizes = {}
            total = 0
            for key, subdata in data.items():
                subsizes[key] = len(str(subdata))
                total += subsizes[key]

            order = sorted(
                subsizes.items(),
                key=lambda item: item[1],
                reverse=True)

            remaining = total
            index = 0
            large_keys = []
            while remaining > limit and index < len(order):
                key, subsize = order[index]
                large_keys.append(key)
                remaining -= subsize
                index += 1

            for large_key in large_keys:
                subdata = breakdown_data(
                    limit,
                    data[large_key],
                    path=path + (large_key,),
                    size=subsizes[large_key])

                try:
                    output.extend(subdata)
                except ValueError:
                    print(f'data can not be broken down to size '
                          f'{limit}: {data[large_key]}')

            pruned = {
                key: value
                for key, value in data.items()
                if key not in large_keys}
            output.append((path, pruned))
            return output

        print(f'Data at {path} is too large, skipped: {size} > {limit}')
        return []

    return [(path, data)]


def get_history_data_db(
    history_collection: Collection,
    experiment_id: Any,
    query: Optional[List] = None,
    func_dict: Optional[Dict] = None,
    f: Optional[Callable[..., Any]] = None,
    filters: Optional[Dict] = None,
    start_time: Union[int, MinKey] = MinKey(),
    end_time: Union[int, MaxKey] = MaxKey(),
    cpus: int = 1,
    host: str = 'localhost',
    port: Any = '27017'
) -> Dict[float, Dict]:
    """Query MongoDB for history data.

    Args:
        history_collection: a MongoDB collection
        experiment_id: the experiment id which is being retrieved
        query: a list of tuples pointing to fields within the experiment data.
            In the format: [('path', 'to', 'field1'), ('path', 'to', 'field2')]
        func_dict: a dict which maps the given query paths to a function that
            operates on the retrieved values and returns the results. If None
            then the raw values are returned.
            In the format: {('path', 'to', 'field1'): function}
        f: a function that applies equally to all fields in query. func_dict
            is the recommended approach and takes priority over f.
        filters: MongoDB query arguments to further filter results
            beyond matching the experiment ID.
        start_time: first simulation time to query.
        end_time: last simulation time to query
        cpus: splits query into this many chunks to run in parallel, useful if
            single-threaded query does not saturate I/O (e.g. on Google Cloud)
        host: used if cpus>1 to create MongoClient in parallel processes
        port: used if cpus>1 to create MongoClient in parallel processes
    Returns:
        data (dict)
    """

    experiment_query = {'experiment_id': experiment_id}
    if filters:
        experiment_query.update(filters)

    projection = None
    if query:
        projection = {f"data.{'.'.join(field)}": 1 for field in query}
        projection['data.time'] = 1
        projection['assembly_id'] = 1

    if cpus > 1:
        chunks = get_data_chunks(history_collection, experiment_id, cpus=cpus)
        queries = []
        for chunk in chunks:
            queries.append({
                **experiment_query,
                '_id': {'$gte': chunk[0], '$lt': chunk[1]},
                'data.time': {'$gte': start_time, '$lte': end_time}
            })
        partial_get_query = partial(get_query, projection, host, port)
        with ProcessPoolExecutor(cpus) as executor:
            queried_chunks = executor.map(partial_get_query, queries)
        cursor = itertools.chain.from_iterable(queried_chunks)
    else:
        experiment_query = {
            **experiment_query,
            'data.time': {'$gte': start_time, '$lte': end_time}
        }
        cursor = history_collection.find(experiment_query, projection)
    raw_data = []
    for document in cursor:
        assert document.get('assembly_id'), \
            "all database documents require an assembly_id"
        if ((f is not None) or (func_dict is not None)) and query:
            for field in query:
                if func_dict:  # func_dict takes priority over f
                    func = func_dict.get(field)
                else:
                    func = f

                document["data"] = apply_func(
                    document["data"], field, func)
        raw_data.append(document)

    # re-assemble data
    assembly = assemble_data(raw_data)

    # restructure by time
    data: Dict[float, Any] = {}
    for datum in assembly.values():
        time = datum['time']
        datum = datum.copy()
        datum.pop('_id', None)
        datum.pop('time', None)
        deep_merge_check(
            data,
            {time: datum},
            check_equality=True,
        )

    return data


def get_query(
    projection: dict,
    host: str,
    port: Any,
    query: dict
) -> List:
    """Helper function for parallel queries

    Args:
        projection: a MongoDB projection in dictionary form
        host: used to create new MongoClient for each parallel process
        port: used to create new MongoClient for each parallel process
        query: a MongoDB query in dictionary form
    Returns:
        List of projected documents for given query
    """
    history_collection = get_local_client(host, port, 'simulations').history
    return list(history_collection.find(query, projection,
        hint=HISTORY_INDEXES[1]))


def get_data_chunks(
    history_collection: Collection,
    experiment_id: str,
    start_time: Union[int, MinKey] = MinKey(),
    end_time: Union[int, MaxKey] = MaxKey(),
    cpus: int = 8
) -> List:
    """Helper function to get chunks for parallel queries

    Args:
        history_collection:`pymongo.Collection`: the MongoDB history collection to query.
        experiment_id:`str`: the experiment id which is being retrieved.
        start_time:`Union[int, MinKey]`: first simulation time to query.
        end_time:`Union[int, MaxKey]`: last simulation time to query.
        cpus:`int`: number of chunks to create.
    Returns:
        List of ObjectId tuples that represent chunk boundaries.
        For each tuple, include ``{'_id': {$gte: tuple[0], $lt: tuple[1]}}``
        in the query to search its corresponding chunk.
    """
    id_cutoffs = list(history_collection.aggregate([{
        '$match': {
            'experiment_id': experiment_id,
            'data.time': {'$gte': start_time, '$lte': end_time}}},
        {'$project': {'_id':1}},
        {'$bucketAuto': {'groupBy': '$_id', 'buckets': cpus}},
        {'$group': {'_id': '', 'splitPoints': {'$push': '$_id.min'}}},
        {'$unset': '_id'}],
        hint={'experiment_id':1, 'data.time':1, '_id':1}))[0]['splitPoints']
    id_ranges = []
    for i in range(len(id_cutoffs)-1):
        id_ranges.append((id_cutoffs[i], id_cutoffs[i+1]))
    id_ranges.append((id_cutoffs[-1], MaxKey()))
    return id_ranges


def deep_merge_check(dct, merge_dct, check_equality=False, path=tuple()):
    """Recursively merge dictionaries with checks to avoid overwriting.

    Args:
        dct: The dictionary to merge into. This dictionary is mutated
            and ends up being the merged dictionary.  If you want to
            keep dct you could call it like
            ``deep_merge_check(copy.deepcopy(dct), merge_dct)``.
        merge_dct: The dictionary to merge into ``dct``.
        check_equality: Whether to use ``==`` to check for conflicts
            instead of the default ``is`` comparator. Note that ``==``
            can cause problems when used with Numpy arrays.
        path: If the ``dct`` is nested within a larger dictionary, the
            path to ``dct``. This is normally an empty tuple (the
            default) for the end user but is used for recursive calls.

    Returns:
        ``dct``

    Raises:
        ValueError: Raised when conflicting values are found between
            ``dct`` and ``merge_dct``.
    """
    for k in merge_dct:
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge_check(dct[k], merge_dct[k], check_equality, path + (k,))
        elif k in dct and not check_equality and (dct[k] is not merge_dct[k]):
            raise ValueError(
                f'Failure to deep-merge dictionaries at path {path + (k,)}: '
                f'{dct[k]} IS NOT {merge_dct[k]}'
            )
        elif k in dct and check_equality and (dct[k] != merge_dct[k]):
            raise ValueError(
                f'Failure to deep-merge dictionaries at path {path + (k,)}: '
                f'{dct[k]} DOES NOT EQUAL {merge_dct[k]}'
            )
        else:
            dct[k] = merge_dct[k]
    return dct


def apply_func(
    document: Any,
    field: Tuple,
    f: Optional[Callable[..., Any]] = None,
) -> Any:
    if field[0] not in document:
        return document
    if len(field) != 1:
        document[field[0]] = apply_func(document[field[0]], field[1:], f)
    elif f is not None:
        document[field[0]] = f(document[field[0]])
    return document


def assemble_data(data: list) -> dict:
    """re-assemble data"""
    assembly: dict = {}
    for datum in data:
        if 'assembly_id' in datum:
            assembly_id = datum['assembly_id']
            if assembly_id not in assembly:
                assembly[assembly_id] = {}
            deep_merge_check(
                assembly[assembly_id],
                datum['data'],
                check_equality=True,
            )
        else:
            assembly_id = str(uuid.uuid4())
            assembly[assembly_id] = datum['data']
    return assembly


def get_local_client(host: str, port: Any, database_name: str) -> Any:
    """Open a MongoDB client onto the given host, port, and DB."""
    client: MongoClient = MongoClient('{}:{}'.format(host, port))
    return client[database_name]
