"""Engine-level observability: structured events, spans and pluggable sinks.

Every ``Composite.run()`` on this engine can describe itself -- run start/end,
a wall-clock-throttled heartbeat, structural changes (a process added or
removed), and, when a process or step raises, *which* process, at what
``global_time``, with a compact summary of the state it was handed. Process
and step authors write nothing for this; the engine emits the events.

The design rules (they are load-bearing, keep them):

* **Never raises into the simulation.** Every sink call is guarded; a sink that
  raises is disabled after one ``sink_error`` event. Enabling instrumentation
  must not change simulation results, only add output.
* **Infrastructure-agnostic.** This module imports no cloud SDK. It ships two
  sinks -- stdout and a local file. Anything else (object stores, HTTP, OTLP,
  ...) is a plugin registered by the caller, an entry point, or a
  ``module:attr`` spec.
* **Off by default in the library, stdout by default in the CLI entrypoints**
  (``run_composite`` / ``run_step``), so a container gets events with zero
  configuration and an importing program gets nothing unless it asks.
* **OpenTelemetry-shaped context, no OpenTelemetry SDK.** Every event carries
  ``trace_id`` / ``span_id`` / ``parent_span_id``; spans nest (task -> run ->
  optional tick/invoke); context is propagated to child processes through a
  W3C ``traceparent`` string. An OTLP exporter is one more sink, later.
* **The engine knows no domain.** The only fields it interprets are its own
  (times, ids, the event name). Everything a CALLER wants to say about a run --
  which experiment, which replicate, which parameter set -- travels in
  ``baggage``, an opaque map with W3C ``baggage`` semantics that the engine
  copies onto every event and never reads. It is string-to-string on the
  wire (``PBG_TRACE_BAGGAGE``), but ``bind()`` stores values as given -- a
  caller may bind an int -- and the engine coerces in neither direction, so a
  consumer must not assume every value is a string. Domain identifiers belong
  in ``baggage``, never in this module's schema or code.
* **Components and dotted event names.** Every event names the ``component``
  that emitted it (a free-form string; this module emits ``"process_bigraph"``)
  and a dotted ``event`` name. The engine's own events are the closed set
  below; callers use their own dotted namespaces and the engine never
  enumerates them.

Switches (all environment variables, all optional)::

    PBG_EVENT_SINKS      comma list of sink specs. ``stdout`` | ``none`` |
                         ``file:<path>`` | ``<scheme>:<rest>`` (registered
                         factory, then the ``process_bigraph.event_sinks``
                         entry-point group) | ``module:attr`` (a callable
                         taking the spec string and returning an EventSink).
                         Unset: the ``default`` the caller passed to
                         ``configure()`` -- 'none' for the library,
                         'stdout' for the CLI entrypoints.
    PBG_TRACEPARENT      W3C ``00-<trace_id>-<span_id>-01`` (the short
                         ``<trace_id>-<span_id>`` form is accepted). Trace ids
                         are whatever the caller derived (e.g. a hash of an
                         upstream correlation id) -- never assumed random.
                         Absent: a fresh trace_id is minted and the first span
                         opened becomes the root.
    PBG_TRACE_BAGGAGE    the caller's context in W3C ``baggage`` form:
                         ``k=v,k2=v2`` (values percent-encoded; no quotes,
                         whitespace, ``$`` or backslashes -- launchers render
                         env through ``docker --env``). A value starting with
                         ``{`` is read as JSON for convenience. Keys and values
                         are opaque to the engine: strings when they arrive
                         this way (W3C baggage is string-to-string on the
                         wire), JSON scalars kept as parsed, and whatever type
                         a caller passes to ``emitter.bind(**kv)`` at runtime.
                         The engine never coerces; consumers coerce and must
                         not assume strings.
    PBG_EVENT_TAGS       same ``key=value,...`` (or JSON) form, copied verbatim
                         onto every event's ``tags`` (job ids, backend names --
                         infrastructure identifiers live here, never in the
                         core schema).
    PBG_EVENT_HEARTBEAT_S  seconds of wall clock between ``tick`` events
                         (default 30; 0 = every tick).
    PBG_EVENT_DETAIL     comma list of ``timing`` (per-process invoke time in
                         ``run.end``), ``invoke`` (one ``process.invoke`` event
                         per process/step call -- large), ``spans`` (a span per
                         ``Composite.run``; always on for the task span).
    PBG_EVENT_SOURCE     label distinguishing concurrent writers of one run
                         (default ``<hostname>-<pid>``).

Deprecated aliases, honoured for one release: ``PROCESS_BIGRAPH_TRACE_FILE=<p>``
== ``PBG_EVENT_SINKS+=file:<p>`` + ``PBG_EVENT_DETAIL+=invoke``;
``PROCESS_BIGRAPH_PROFILE_PROCESSES=1`` == ``PBG_EVENT_DETAIL+=timing``.

Event schema (one JSON object per line, ``default=str`` serialisation)::

    {"v": 1, "ts": "<UTC ISO 8601>", "seq": <per-process counter>,
     "source": "<host-pid>", "component": "process_bigraph",
     "event": "<dotted name>", "level": "debug|info|warning|error",
     "trace_id": "<32 hex>", "span_id": "<16 hex>", "parent_span_id": "<16 hex>|null",
     "global_time": <float|null>, "wall_time": <seconds since configure()>,
     "baggage": {<caller's opaque context>}, "tags": {...}, "payload": {...}}

The engine's own events (callers use their own dotted namespaces)::

    run.start / run.end      Composite.run (span ``run`` when ``spans`` detail is on).
                             Rate-limited per Composite: the first run and any
                             run ending in error always emit; otherwise at most
                             one pair per ``heartbeat_s``, with the runs skipped
                             in between folded into the next ``run.end`` as
                             ``runs=N`` and summed ``total``/``process_time``/
                             ``framework_time``/``ticks``. An error whose start
                             was suppressed carries ``start_suppressed``.
    tick                     the throttled heartbeat inside the run loop
    structure.changed        a reconcile reported a structural change
    process.exception        a Process/Step invoke raised (path, class, address,
                             interval, global_time, state summary)
    process.init             a protocol runtime initialised a remote process
    runtime.error            a protocol runtime flush failed
    task.start / task.end    the CLI entrypoints run_composite / run_step (span ``task``)
    span.start / span.end    every span boundary (span.end repeats start_ts and
                             carries duration_s and status)
    sink.error               a sink raised and was disabled
    process.invoke           opt-in: one record per invoke
    process.timing           opt-in: per-process invoke time (inside run.end)
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import importlib
import json
import os
import secrets
import socket
import sys
import threading
import time as _time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

SCHEMA_VERSION = 1
ENGINE_COMPONENT = 'process_bigraph'
ENTRY_POINT_GROUP = 'process_bigraph.event_sinks'
DEFAULT_HEARTBEAT_S = 30.0
LEVELS = ('debug', 'info', 'warning', 'error')


# ---------------------------------------------------------------------------
# State summaries (shared by the exception hook, invoke tracing, and runners)
# ---------------------------------------------------------------------------

def _summarize_value(value, depth=0):
    """Lightweight, JSON-safe summary of a state/update fragment.

    - Scalars and short strings inlined.
    - Numpy arrays as ``{shape, dtype, sum, min, max, nan_count, neg_count,
      head}`` (the last four only for numeric dtypes).
    - Dicts recursed (capped depth and width).
    - Lists/tuples shown as their first few items.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 3:
        return f'<{type(value).__name__}>'
    if isinstance(value, np.ndarray):
        try:
            out: Dict[str, Any] = {
                '_np': True,
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'sum': None,
                'head': value.flatten()[:5].tolist() if value.size else [],
            }
            if value.dtype.kind in 'fi' and value.size:
                flat = value.ravel()
                out['sum'] = float(np.nansum(flat))
                out['min'] = float(np.nanmin(flat))
                out['max'] = float(np.nanmax(flat))
                out['nan_count'] = int(np.isnan(flat).sum()) if value.dtype.kind == 'f' else 0
                out['neg_count'] = int((flat < 0).sum())
            return out
        except Exception:
            return f'<ndarray shape={getattr(value, "shape", "?")}>'
    if isinstance(value, dict):
        return {k: _summarize_value(v, depth + 1) for k, v in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v, depth + 1) for v in value[:10]]
    return f'<{type(value).__name__}>'


def summarize_state(state, max_roots: int = 64) -> Any:
    """Summarise one level of root stores (what a process was handed).

    Each root store gets ``_summarize_value`` at depth 2, i.e. at most one
    nested level of dict keys -- enough to see counts, sizes and NaN/negative
    flags without walking a whole state tree. Never raises."""
    try:
        if isinstance(state, dict):
            return {k: _summarize_value(v, 2) for k, v in list(state.items())[:max_roots]}
        return _summarize_value(state)
    except Exception as exc:  # pragma: no cover - defensive
        return f'<unsummarizable: {exc!r}>'


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class EventSink(ABC):
    """Where events go. Implementations must be cheap and must not block the
    simulation; a sink that raises is disabled by the emitter."""

    @abstractmethod
    def emit(self, event: Dict[str, Any]) -> None: ...

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class NullSink(EventSink):
    def emit(self, event):
        return None


class StdoutSink(EventSink):
    """One JSON line per event on ``sys.stdout`` (resolved at emit time so
    captured/redirected stdout is honoured)."""

    def emit(self, event):
        sys.stdout.write(json.dumps(event, default=str) + '\n')
        sys.stdout.flush()

    def flush(self):
        try:
            sys.stdout.flush()
        except Exception:
            pass


class FileSink(EventSink):
    """Line-buffered append to a local file."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, 'a', buffering=1)

    def emit(self, event):
        self._fh.write(json.dumps(event, default=str) + '\n')

    def flush(self):
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class MultiSink(EventSink):
    def __init__(self, sinks: Iterable[EventSink]):
        self.sinks = list(sinks)

    def emit(self, event):
        for sink in self.sinks:
            sink.emit(event)

    def flush(self):
        for sink in self.sinks:
            sink.flush()

    def close(self):
        for sink in self.sinks:
            sink.close()


_SINK_FACTORIES: Dict[str, Callable[[str], EventSink]] = {}


def register_sink_factory(scheme: str, factory: Callable[[str], EventSink]) -> None:
    """Register ``factory(spec) -> EventSink`` for specs starting with
    ``<scheme>:``. The factory receives the full spec string (e.g.
    ``mysink://host/path``)."""
    if not callable(factory):
        raise TypeError(f'sink factory for {scheme!r} is not callable: {factory!r}')
    _SINK_FACTORIES[scheme] = factory


def _entry_point_factory(scheme: str) -> Optional[Callable[[str], EventSink]]:
    try:
        from importlib import metadata
        eps = metadata.entry_points()
        group = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, 'select') \
            else eps.get(ENTRY_POINT_GROUP, [])
        for ep in group:
            if ep.name == scheme:
                return ep.load()
    except Exception:
        return None
    return None


def resolve_sink(spec: str) -> Optional[EventSink]:
    """Turn one spec string into a sink, or ``None`` (with one warning) when it
    cannot be resolved. Never raises."""
    spec = (spec or '').strip()
    if not spec or spec == 'none':
        return None
    try:
        if spec == 'stdout':
            return StdoutSink()
        if spec.startswith('file:'):
            return FileSink(spec[len('file:'):])
        if ':' in spec:
            scheme, rest = spec.split(':', 1)
            factory = _SINK_FACTORIES.get(scheme) or _entry_point_factory(scheme)
            if factory is not None:
                return factory(spec)
            # module:attr -- a callable taking the spec string
            if rest.isidentifier():
                try:
                    module = importlib.import_module(scheme)
                except ImportError:
                    module = None
                if module is not None:
                    target = getattr(module, rest)
                    sink = target(spec) if callable(target) else target
                    if isinstance(sink, EventSink):
                        return sink
                    raise TypeError(f'{spec!r} did not produce an EventSink')
        raise ValueError(f'unknown event sink spec {spec!r}')
    except Exception as exc:
        warnings.warn(f'process_bigraph.events: dropping sink {spec!r}: {exc!r}')
        return None


def resolve_sinks(specs: str) -> List[EventSink]:
    sinks = []
    for spec in (specs or '').split(','):
        sink = resolve_sink(spec)
        if sink is not None:
            sinks.append(sink)
    return sinks


# ---------------------------------------------------------------------------
# Trace / span context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    def traceparent(self) -> str:
        return f'00-{self.trace_id}-{self.span_id}-01'


_current_span: contextvars.ContextVar[Optional[SpanContext]] = contextvars.ContextVar(
    'process_bigraph_current_span', default=None)


def mint_trace_id() -> str:
    return secrets.token_hex(16)


def mint_span_id() -> str:
    return secrets.token_hex(8)


def parse_traceparent(value: Optional[str]):
    """``00-<32hex>-<16hex>-<flags>`` or ``<32hex>-<16hex>`` -> (trace_id,
    span_id), else ``None``."""
    if not value:
        return None
    parts = value.strip().split('-')
    if len(parts) == 4:
        parts = parts[1:3]
    if len(parts) < 2:
        return None
    trace_id, span_id = parts[0].lower(), parts[1].lower()
    if len(trace_id) != 32 or len(span_id) != 16:
        return None
    try:
        int(trace_id, 16)
        int(span_id, 16)
    except ValueError:
        return None
    return trace_id, span_id


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


class Span:
    """An open span. ``end()`` emits ``span_end`` and restores the parent
    context. Use ``EventEmitter.span()`` for the context-manager form."""

    def __init__(self, emitter: 'EventEmitter', name: str, ctx: SpanContext,
                 attrs: Dict[str, Any], token):
        self.emitter = emitter
        self.name = name
        self.ctx = ctx
        self.attrs = attrs
        self.start_ts = _utc_now()
        self._start_mono = _time.monotonic()
        self._token = token
        self._ended = False

    def end(self, status: str = 'ok', error: Optional[str] = None) -> None:
        if self._ended:
            return
        self._ended = True
        duration = _time.monotonic() - self._start_mono
        self.emitter._emit_raw(
            'span.end', 'error' if status == 'error' else 'info', ENGINE_COMPONENT,
            {'name': self.name, 'attrs': self.attrs, 'start_ts': self.start_ts,
             'end_ts': _utc_now(), 'duration_s': round(duration, 6),
             'status': status, 'error': error},
            ctx=self.ctx)
        try:
            _current_span.reset(self._token)
        except (ValueError, LookupError):
            # Ended from a different context (thread/executor); fall back to
            # restoring the parent explicitly.
            parent = None
            if self.ctx.parent_span_id is not None:
                parent = SpanContext(self.ctx.trace_id, self.ctx.parent_span_id, None)
            _current_span.set(parent)


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------

class EventEmitter:
    """Process-wide event emitter. Cheap when nothing is configured: every
    public method short-circuits on ``enabled``."""

    def __init__(self, sinks: Optional[Iterable[EventSink]] = None, *,
                 baggage: Optional[Dict[str, Any]] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 heartbeat_s: float = DEFAULT_HEARTBEAT_S,
                 detail: Iterable[str] = (),
                 trace_id: Optional[str] = None,
                 root_span_id: Optional[str] = None,
                 source: Optional[str] = None):
        self._sinks: List[EventSink] = list(sinks or [])
        # Opaque to the engine: copied onto every event, never interpreted.
        self.baggage: Dict[str, Any] = dict(baggage or {})
        self.tags: Dict[str, Any] = dict(tags or {})
        self.heartbeat_s = float(heartbeat_s)
        detail = {d.strip() for d in detail if d and d.strip()}
        self.detail_timing = 'timing' in detail
        self.detail_invoke = 'invoke' in detail
        self.detail_spans = 'spans' in detail
        self.trace_id = trace_id or mint_trace_id()
        self.root_span_id = root_span_id      # the parent handed to us, if any
        self.source = source or f'{socket.gethostname()}-{os.getpid()}'
        self._lock = threading.Lock()
        self._seq = 0
        self._origin = _time.monotonic()
        self._last_heartbeat: Optional[float] = None
        self._counters: Dict[str, float] = {}
        self._ticks_since = 0
        self._ticks_total = 0
        self._in_sink_error = False

    # -- configuration -------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    def add_sink(self, sink: EventSink) -> 'EventEmitter':
        """Attach one more sink to a live emitter (idempotent per instance).
        The public way for an embedder to add a destination after
        ``configure()`` -- e.g. a file next to a task's output -- without
        reaching into the emitter's internals."""
        with self._lock:
            if not any(existing is sink for existing in self._sinks):
                self._sinks.append(sink)
        return self

    def remove_sink(self, sink: EventSink) -> 'EventEmitter':
        """Detach a sink added earlier (no-op when absent). The sink is not
        closed; the caller owns its lifetime."""
        with self._lock:
            self._sinks = [existing for existing in self._sinks if existing is not sink]
        return self

    def bind(self, **baggage) -> 'EventEmitter':
        """Add to (or overwrite in) the opaque baggage copied onto every
        event. Any key; a value of ``None`` removes the key. Values are stored
        as given (not coerced to ``str``), so an event's ``baggage`` may mix
        strings from ``PBG_TRACE_BAGGAGE`` with whatever a caller bound."""
        for key, value in baggage.items():
            if value is None:
                self.baggage.pop(key, None)
            else:
                self.baggage[key] = value
        return self

    # -- context -------------------------------------------------------- #

    def current_context(self) -> Optional[SpanContext]:
        return _current_span.get()

    def current_traceparent(self) -> str:
        ctx = _current_span.get()
        if ctx is not None:
            return ctx.traceparent()
        return SpanContext(self.trace_id, self.root_span_id or '0' * 16).traceparent()

    def start_span(self, span_name: str, /, **attrs) -> Span:
        """Open a child of the current span (or of the traceparent root).
        ``attrs`` are free-form and may include a key called ``name``."""
        parent = _current_span.get()
        parent_id = parent.span_id if parent is not None else self.root_span_id
        ctx = SpanContext(self.trace_id, mint_span_id(), parent_id)
        token = _current_span.set(ctx)
        span = Span(self, span_name, ctx, dict(attrs), token)
        self._emit_raw('span.start', 'info', ENGINE_COMPONENT,
                       {'name': span_name, 'attrs': dict(attrs), 'start_ts': span.start_ts},
                       ctx=ctx)
        return span

    @contextlib.contextmanager
    def span(self, span_name: str, /, **attrs):
        span = self.start_span(span_name, **attrs)
        try:
            yield span
        except BaseException as exc:
            span.end('error', f'{type(exc).__name__}: {exc}')
            raise
        else:
            span.end('ok')

    # -- events --------------------------------------------------------- #

    def event(self, event_name: str, /, level: str = 'info',
              component: str = ENGINE_COMPONENT, global_time=None, **payload) -> None:
        """Emit one event. ``event_name`` is a dotted name in the caller's
        namespace; ``component`` names the emitting code (free-form string);
        ``global_time`` lands in the top-level field; everything else (any
        key, including ``name``) in ``payload``."""
        if not self._sinks:
            return
        self._emit_raw(event_name, level, component, payload, global_time=global_time)

    def count(self, **counters) -> None:
        """Accumulate per-heartbeat counters without emitting."""
        if not self._sinks:
            return
        for key, value in counters.items():
            self._counters[key] = self._counters.get(key, 0) + value

    def heartbeat(self, global_time=None, **counters) -> bool:
        """Emit a ``tick`` at most every ``heartbeat_s`` seconds of wall
        clock. Returns True when an event was emitted."""
        if not self._sinks:
            return False
        self._ticks_since += 1
        self._ticks_total += 1
        now = _time.monotonic()
        if self._last_heartbeat is not None and now - self._last_heartbeat < self.heartbeat_s:
            return False
        self._last_heartbeat = now
        payload = dict(self._counters)
        payload.update(counters)
        payload['ticks'] = self._ticks_since
        payload['ticks_total'] = self._ticks_total
        self._counters = {}
        self._ticks_since = 0
        self._emit_raw('tick', 'debug', ENGINE_COMPONENT, payload, global_time=global_time)
        return True

    def reset_run_counters(self) -> None:
        self._counters = {}
        self._ticks_since = 0
        self._ticks_total = 0
        self._last_heartbeat = None

    @property
    def ticks_total(self) -> int:
        return self._ticks_total

    def exception(self, exc: BaseException, *, event_name: str = 'process.exception',
                  **ctx) -> None:
        """Record an exception with its engine context as ``event_name``
        (``process.exception`` for a raising invoke, ``runtime.error`` for a
        protocol-runtime flush). Attaches ``exc.pbg_context`` (innermost hook
        wins) and a note on 3.11+; the caller re-raises the original exception
        unchanged."""
        context = {'exc_type': type(exc).__name__, 'exc_msg': str(exc)[:2000]}
        context.update(ctx)
        try:
            if not hasattr(exc, 'pbg_context'):
                exc.pbg_context = context
                if hasattr(exc, 'add_note'):
                    where = context.get('path') or context.get('runtime') or '?'
                    exc.add_note(f"[process_bigraph] raised in {where} "
                                 f"at global_time={context.get('global_time')}")
        except Exception:
            pass
        if not self._sinks:
            return
        self._emit_raw(event_name, 'error', ENGINE_COMPONENT, context,
                       global_time=ctx.get('global_time'))

    def invoke(self, path, instance, state, interval, update) -> None:
        """Opt-in per-invoke record (``PBG_EVENT_DETAIL=invoke``)."""
        if not self._sinks or not self.detail_invoke:
            return
        gt = state.get('global_time') if isinstance(state, dict) else None
        self._emit_raw('process.invoke', 'debug', ENGINE_COMPONENT, {
            'path': '/'.join(str(p) for p in path) if isinstance(path, (list, tuple)) else str(path),
            'cls': type(instance).__name__,
            'interval': interval,
            'input': _summarize_value(state),
            'output': _summarize_value(update),
        }, global_time=gt)

    def flush(self) -> None:
        for sink in list(self._sinks):
            try:
                sink.flush()
            except Exception:
                pass

    def close(self) -> None:
        for sink in list(self._sinks):
            try:
                sink.close()
            except Exception:
                pass

    # -- internals ------------------------------------------------------ #

    def _build(self, name, level, component, payload, *, ctx=None, global_time=None):
        ctx = ctx or _current_span.get()
        with self._lock:
            self._seq += 1
            seq = self._seq
        return {
            'v': SCHEMA_VERSION,
            'ts': _utc_now(),
            'seq': seq,
            'component': str(component),
            'event': name,
            'level': level if level in LEVELS else 'info',
            'trace_id': ctx.trace_id if ctx else self.trace_id,
            'span_id': ctx.span_id if ctx else None,
            'parent_span_id': ctx.parent_span_id if ctx else self.root_span_id,
            'global_time': global_time,
            'wall_time': round(_time.monotonic() - self._origin, 3),
            'source': self.source,
            'baggage': dict(self.baggage),
            'tags': self.tags,
            'payload': payload,
        }

    def _emit_raw(self, name, level, component, payload, *, ctx=None, global_time=None) -> None:
        if not self._sinks:
            return
        try:
            event = self._build(name, level, component, payload, ctx=ctx, global_time=global_time)
        except Exception:
            return
        self._dispatch(event)

    def _dispatch(self, event) -> None:
        for sink in list(self._sinks):
            try:
                sink.emit(event)
            except Exception as exc:
                # Disable the sink, tell the others once, never raise.
                try:
                    self._sinks.remove(sink)
                except ValueError:
                    pass
                if not self._in_sink_error:
                    self._in_sink_error = True
                    try:
                        self._emit_raw('sink.error', 'error', ENGINE_COMPONENT,
                                       {'sink': type(sink).__name__, 'error': repr(exc)})
                    finally:
                        self._in_sink_error = False


# ---------------------------------------------------------------------------
# Process-wide configuration
# ---------------------------------------------------------------------------

_EMITTER: Optional[EventEmitter] = None
_EMITTER_LOCK = threading.RLock()   # re-entrant: get_emitter() -> configure() nests


def parse_baggage(raw: Optional[str]) -> Dict[str, Any]:
    """W3C ``baggage`` form ``k=v,k2=v2`` (values percent-encoded), or a JSON
    object when the value starts with ``{``. Keys and values are opaque to
    the engine: no coercion (W3C baggage is string-to-string on the wire;
    JSON scalars are kept as parsed). Never raises; malformed entries are
    skipped."""
    if not raw or not raw.strip():
        return {}
    raw = raw.strip()
    out: Dict[str, Any] = {}
    if raw.startswith('{'):
        try:
            value = json.loads(raw)
            out = dict(value) if isinstance(value, dict) else {}
        except Exception:
            warnings.warn('process_bigraph.events: baggage looks like JSON but does not parse; ignored')
            return {}
    else:
        from urllib.parse import unquote
        for item in raw.split(','):
            if '=' not in item:
                continue
            key, value = item.split('=', 1)
            key = key.strip()
            if not key:
                continue
            # W3C baggage allows ``;``-separated properties after the value
            out[key] = unquote(value.split(';', 1)[0].strip())
    return out


def configure(spec: Optional[str] = None, *, default: str = 'none',
              env: Optional[Dict[str, str]] = None,
              sinks: Optional[Iterable[EventSink]] = None) -> EventEmitter:
    """Build (and install as the process-wide emitter) from ``spec`` or the
    environment. ``default`` is the sink spec used when neither ``spec`` nor
    ``PBG_EVENT_SINKS`` is set."""
    global _EMITTER
    env = os.environ if env is None else env

    sink_spec = spec if spec is not None else env.get('PBG_EVENT_SINKS')
    if sink_spec is None:
        sink_spec = default
    detail = {d.strip() for d in env.get('PBG_EVENT_DETAIL', '').split(',') if d.strip()}

    # Deprecated aliases.
    legacy_trace = env.get('PROCESS_BIGRAPH_TRACE_FILE')
    if legacy_trace:
        sink_spec = f'{sink_spec},file:{legacy_trace}' if sink_spec and sink_spec != 'none' \
            else f'file:{legacy_trace}'
        detail.add('invoke')
    if env.get('PROCESS_BIGRAPH_PROFILE_PROCESSES'):
        detail.add('timing')

    resolved = list(sinks or [])
    resolved.extend(resolve_sinks(sink_spec))

    parsed = parse_traceparent(env.get('PBG_TRACEPARENT'))
    trace_id, root_span = parsed if parsed else (None, None)
    baggage = parse_baggage(env.get('PBG_TRACE_BAGGAGE'))
    tags = parse_baggage(env.get('PBG_EVENT_TAGS'))
    try:
        heartbeat_s = float(env.get('PBG_EVENT_HEARTBEAT_S', DEFAULT_HEARTBEAT_S))
    except ValueError:
        heartbeat_s = DEFAULT_HEARTBEAT_S

    emitter = EventEmitter(resolved, baggage=baggage, tags=tags,
                           heartbeat_s=heartbeat_s, detail=detail,
                           trace_id=trace_id, root_span_id=root_span,
                           source=env.get('PBG_EVENT_SOURCE'))
    with _EMITTER_LOCK:
        _EMITTER = emitter
    return emitter


def get_emitter() -> EventEmitter:
    """The process-wide emitter, lazily configured with ``default='none'``
    (i.e. silent) when nothing configured one."""
    global _EMITTER
    if _EMITTER is None:
        with _EMITTER_LOCK:
            if _EMITTER is None:
                _EMITTER = configure(default='none')
    return _EMITTER


def set_emitter(emitter: Optional[EventEmitter]) -> None:
    """Install a prebuilt emitter (tests, embedders)."""
    global _EMITTER
    with _EMITTER_LOCK:
        _EMITTER = emitter


def traceback_tail(exc: BaseException, lines: int = 40) -> str:
    text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return '\n'.join(text.splitlines()[-lines:])


def exception_record(exc: BaseException, **extra) -> Dict[str, Any]:
    """A JSON-safe record describing an exception that ended a run or task,
    including the engine context attached by ``EventEmitter.exception`` and
    the emitter's current baggage."""
    record: Dict[str, Any] = {
        'exc_type': type(exc).__name__,
        'exc_msg': str(exc)[:2000],
        'traceback_tail': traceback_tail(exc),
        'pbg_context': getattr(exc, 'pbg_context', None),
        'baggage': dict(get_emitter().baggage),
        'ts': _utc_now(),
    }
    record.update(extra)
    return record


__all__ = [
    'EventSink', 'NullSink', 'StdoutSink', 'FileSink', 'MultiSink',
    'register_sink_factory', 'resolve_sink', 'resolve_sinks',
    'SpanContext', 'Span', 'EventEmitter', 'configure', 'get_emitter',
    'set_emitter', 'parse_traceparent', 'parse_baggage', 'mint_trace_id', 'mint_span_id',
    'summarize_state', 'exception_record', 'traceback_tail', 'SCHEMA_VERSION',
    'ENGINE_COMPONENT',
]
