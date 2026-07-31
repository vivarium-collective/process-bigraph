"""
===============================================================
The ``git:`` / remote protocol — resolve a repo address to an Edge
===============================================================

A ``git:`` address resolves a *repository* to a runnable process ``Edge``:

    git:<owner>/<repo>@<ref>#<module>:<callable>

e.g. ``git:CovertLab/vEcoli@main#vecoli.workflow:make_process``. ``@<ref>`` is
optional (defaults to the repo's default branch); ``#<module>:<callable>`` is
required and names the entrypoint the repo exposes.

Resolution (spec §1):

  1. **Fetch + pin** the repo at ``<ref>`` into a content-addressed cache and
     record the resolved **commit SHA** — a moving ref re-resolves; a SHA is
     frozen (the reproducibility unit).
  2. **Materialize a venv** for that checkout (``uv venv`` + install the repo),
     keyed by SHA. The foreign stack lives only in that venv.
  3. **Expose ``<module>:<callable>``** as an ``Edge`` whose ``update()`` proxies
     over a small stdio-RPC boundary to a subprocess running the entrypoint in
     the repo's own venv (``_git_worker.py``). No foreign code is imported into
     the host interpreter (D1(a) = subprocess + per-SHA venv).

Trust (D2): a ``git:`` address whose ``owner/repo`` is not on the **allow-list**
is refused, not run. The allow-list defaults to ``CovertLab/vEcoli`` and is
supplied by the caller (``set_allow_list`` / ``add_allowed_repo``) — pbg stays
workspace-agnostic and never reads ``workspace.yaml`` itself.

Conformance (D5): :func:`check_conformance` verifies the resolved
``interface()`` admits a declared face and *names* any mismatch. A run never
starts against a non-conforming address (contract 2).
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import shutil
import hashlib
import pathlib
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Callable

from bigraph_schema.schema import Protocol, String
from bigraph_schema.methods import load_protocol

from process_bigraph.composite import Process


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class GitProtocolError(Exception):
    """Any failure resolving or running a ``git:`` address. Raised as a clear
    error (contract 3: a fetched-code failure degrades to an error, not a
    host crash)."""


class AllowListError(GitProtocolError):
    """An ``owner/repo`` outside the allow-list was addressed (D2, contract 4)."""


class ConformanceError(GitProtocolError):
    """The resolved ``interface()`` does not admit the declared face
    (D5, contract 2)."""


# ---------------------------------------------------------------------------
# Address grammar
# ---------------------------------------------------------------------------
_ADDRESS_RE = re.compile(
    r'^(?P<owner>[^/@#]+)/(?P<repo>[^/@#]+)'
    r'(?:@(?P<ref>[^#]+))?'
    r'#(?P<module>[A-Za-z_][\w.]*):(?P<callable>[A-Za-z_][\w.]*)$'
)

# The default branch sentinel — resolved against the remote's HEAD when no
# ``@ref`` is given.
DEFAULT_REF = 'HEAD'


@dataclass(frozen=True)
class GitAddress:
    """Parsed ``git:`` address. ``repo_slug`` is the allow-list key."""
    owner: str
    repo: str
    ref: str
    module: str
    entry: str

    @property
    def repo_slug(self) -> str:
        return f'{self.owner}/{self.repo}'

    def __str__(self) -> str:
        return (f'git:{self.owner}/{self.repo}@{self.ref}'
                f'#{self.module}:{self.entry}')


def parse_git_address(data: str) -> GitAddress:
    """Parse the ``data`` half of a ``git:`` address (everything after
    ``git:``) into a :class:`GitAddress`. ``normalize_address`` splits on the
    first ``:`` only, so the ``module:callable`` colon survives inside
    ``data``."""
    if not isinstance(data, str):
        raise GitProtocolError(f'git address must be a string, not {data!r}')
    match = _ADDRESS_RE.match(data.strip())
    if match is None:
        raise GitProtocolError(
            f'malformed git address {data!r}. Expected '
            f'"<owner>/<repo>[@<ref>]#<module>:<callable>", e.g. '
            f'"CovertLab/vEcoli@main#vecoli.workflow:make_process".')
    groups = match.groupdict()
    return GitAddress(
        owner=groups['owner'],
        repo=groups['repo'],
        ref=groups['ref'] or DEFAULT_REF,
        module=groups['module'],
        entry=groups['callable'])


# ---------------------------------------------------------------------------
# Allow-list (D2) — caller-supplied, pbg stays workspace-agnostic
# ---------------------------------------------------------------------------
_DEFAULT_ALLOW_LIST = frozenset({'CovertLab/vEcoli'})
_ALLOW_LIST: set = set(_DEFAULT_ALLOW_LIST)


def set_allow_list(repos) -> None:
    """Replace the allow-list. The caller (e.g. the workbench, reading
    ``workspace.yaml``) supplies the set of ``owner/repo`` slugs it trusts."""
    global _ALLOW_LIST
    _ALLOW_LIST = set(repos)


def add_allowed_repo(repo_slug: str) -> None:
    """Add one ``owner/repo`` to the allow-list (an explicitly-trusted fork)."""
    _ALLOW_LIST.add(repo_slug)


def get_allow_list() -> set:
    """Return a copy of the current allow-list."""
    return set(_ALLOW_LIST)


def reset_allow_list() -> None:
    """Restore the default allow-list (``CovertLab/vEcoli``). For tests."""
    global _ALLOW_LIST
    _ALLOW_LIST = set(_DEFAULT_ALLOW_LIST)


def check_allow_list(address: GitAddress) -> None:
    if address.repo_slug not in _ALLOW_LIST:
        raise AllowListError(
            f'git address {address.repo_slug!r} is not on the allow-list '
            f'{sorted(_ALLOW_LIST)!r}. Add it explicitly with '
            f'add_allowed_repo({address.repo_slug!r}) — git: refuses to run '
            f'code from un-allow-listed repositories.')


# ---------------------------------------------------------------------------
# Repo URL resolution — a hook so forks / private mirrors / test fixtures can
# override the clone URL for an ``owner/repo`` (default: github.com).
# ---------------------------------------------------------------------------
_URL_OVERRIDES: Dict[str, str] = {}


def set_repo_url(repo_slug: str, url: str) -> None:
    """Point ``owner/repo`` at an explicit git URL (a fork, a local mirror, or
    a ``file://`` fixture in tests). Does *not* implicitly allow-list it."""
    _URL_OVERRIDES[repo_slug] = url


def clear_repo_url(repo_slug: str) -> None:
    _URL_OVERRIDES.pop(repo_slug, None)


def repo_url(address: GitAddress) -> str:
    override = _URL_OVERRIDES.get(address.repo_slug)
    if override:
        return override
    return f'https://github.com/{address.owner}/{address.repo}'


# ---------------------------------------------------------------------------
# Cache / pin (D3)
# ---------------------------------------------------------------------------
def cache_root() -> pathlib.Path:
    root = os.environ.get('PBG_GIT_CACHE') or os.path.join(
        os.path.expanduser('~'), '.process_bigraph', 'git_cache')
    path = pathlib.Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug_dir(address: GitAddress) -> pathlib.Path:
    return cache_root() / f'{address.owner}__{address.repo}'


def _run_git(args, cwd=None) -> str:
    result = subprocess.run(
        ['git', *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    if result.returncode != 0:
        raise GitProtocolError(
            f'git {" ".join(args)} failed (exit {result.returncode}): '
            f'{result.stderr.strip()}')
    return result.stdout


def resolve_sha(address: GitAddress) -> str:
    """Resolve ``address.ref`` to a concrete commit SHA against the remote,
    without cloning (``git ls-remote``). A ref that is already a full 40-hex
    SHA is returned as-is (frozen)."""
    ref = address.ref
    if re.fullmatch(r'[0-9a-f]{40}', ref or ''):
        return ref

    url = repo_url(address)
    if ref == DEFAULT_REF:
        out = _run_git(['ls-remote', '--symref', url, 'HEAD'])
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == 'HEAD':
                return parts[0]
        raise GitProtocolError(
            f'could not resolve default branch (HEAD) of {url}')

    out = _run_git(['ls-remote', url, ref])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        # Might already be a short/long SHA the remote accepts on fetch.
        if re.fullmatch(r'[0-9a-f]{7,40}', ref):
            return ref
        raise GitProtocolError(
            f'ref {ref!r} not found in {url}')
    # Prefer an exact refs/heads or refs/tags match; else first line.
    for line in lines:
        sha, name = line.split('\t', 1) if '\t' in line else line.split()
        if name in (f'refs/heads/{ref}', f'refs/tags/{ref}', ref):
            return sha
    return lines[0].split()[0]


@dataclass
class Pin:
    """A recorded resolution — the reproducibility unit (D3)."""
    repo_slug: str
    url: str
    ref: str
    sha: str
    resolved_at: float

    def to_dict(self) -> dict:
        return asdict(self)


def _sha_dir(address: GitAddress, sha: str) -> pathlib.Path:
    return _slug_dir(address) / sha


def read_pin(address: GitAddress, sha: str) -> Optional[Pin]:
    pin_path = _sha_dir(address, sha) / 'pin.json'
    if pin_path.exists():
        return Pin(**json.loads(pin_path.read_text()))
    return None


def resolve_and_pin(address: GitAddress) -> Pin:
    """Resolve ``ref`` -> SHA and record the pin under the cache. Records the
    SHA; the run uses the recorded SHA (a later ref move re-resolves to a new
    SHA and is surfaced, never silently re-run)."""
    check_allow_list(address)
    sha = resolve_sha(address)
    sha_dir = _sha_dir(address, sha)
    sha_dir.mkdir(parents=True, exist_ok=True)
    pin = read_pin(address, sha)
    if pin is None:
        pin = Pin(
            repo_slug=address.repo_slug,
            url=repo_url(address),
            ref=address.ref,
            sha=sha,
            resolved_at=time.time())
        (sha_dir / 'pin.json').write_text(json.dumps(pin.to_dict(), indent=2))
    return pin


# ---------------------------------------------------------------------------
# Materialization — checkout + per-SHA venv
# ---------------------------------------------------------------------------
@dataclass
class Materialized:
    address: GitAddress
    pin: Pin
    repo_dir: pathlib.Path
    venv_python: pathlib.Path


def _uv() -> str:
    uv = shutil.which('uv')
    if uv is None:
        raise GitProtocolError(
            'the git: protocol needs `uv` to materialize per-SHA venvs; '
            'install it (https://docs.astral.sh/uv/) or add it to PATH.')
    return uv


def _checkout(address: GitAddress, pin: Pin, repo_dir: pathlib.Path) -> None:
    if (repo_dir / '.git').exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    # Full clone then checkout the pinned SHA — robust across servers that
    # don't allow fetching arbitrary SHAs directly. Repos are cached per-SHA
    # so this is paid once.
    _run_git(['clone', '--quiet', pin.url, str(repo_dir)])
    _run_git(['checkout', '--quiet', pin.sha], cwd=str(repo_dir))


def _build_venv(repo_dir: pathlib.Path, venv_dir: pathlib.Path) -> pathlib.Path:
    """``uv venv`` + install the repo into it. Returns the venv python path."""
    python = venv_dir / 'bin' / 'python'
    stamp = venv_dir / '.pbg-installed'
    if stamp.exists() and python.exists():
        return python

    uv = _uv()
    if not python.exists():
        subprocess.run([uv, 'venv', '--quiet', str(venv_dir)],
                       check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Install the repo into its venv, **editable**. A wheel install ships only
    # what the repo declares as packages, which for a scientific repo is
    # routinely incomplete — CovertLab/vEcoli's wheel omits `ecoli.library`
    # and every `.tsv`/`.json` data file its reconstruction reads. The worker
    # cannot fall back to the source tree either: it is launched as
    # `python _git_worker.py`, so `sys.path[0]` is the *worker's* directory,
    # not `cwd`. Editable makes the checkout itself importable, data and all.
    result = subprocess.run(
        [uv, 'pip', 'install', '--python', str(python), '--quiet', '-e', '.'],
        cwd=str(repo_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise GitProtocolError(
            f'failed to install repo into venv (uv pip install .): '
            f'{result.stderr.strip()}')
    stamp.write_text('ok')
    return python


def materialize(address: GitAddress) -> Materialized:
    """Full resolve -> pin -> checkout -> venv path. Cheap on a warm cache."""
    pin = resolve_and_pin(address)
    sha_dir = _sha_dir(address, pin.sha)
    repo_dir = sha_dir / 'repo'
    venv_dir = sha_dir / 'venv'
    _checkout(address, pin, repo_dir)
    venv_python = _build_venv(repo_dir, venv_dir)
    return Materialized(
        address=address,
        pin=pin,
        repo_dir=repo_dir,
        venv_python=venv_python)


# ---------------------------------------------------------------------------
# The remote edge — a thin stdio-RPC proxy over the venv subprocess
# ---------------------------------------------------------------------------
_WORKER_PATH = str(pathlib.Path(__file__).with_name('_git_worker.py'))


class GitRemoteProcess(Process):
    """A ``Process`` whose ``update()`` proxies over stdio to a subprocess
    running the repo's entrypoint in the repo's *own* venv.

    Bound per-address by :func:`load_protocol` (subclasses carry the
    materialization as class attributes, mirroring the ray protocol)."""

    # Populated on the per-address bound subclass.
    _materialized: Optional[Materialized] = None

    config_schema: Any = {}

    def initialize(self, config):
        mat = self._materialized
        if mat is None:
            raise GitProtocolError(
                'GitRemoteProcess used without a bound address; go through '
                'the git: protocol (load_protocol).')
        self._ended = False
        self._proc = subprocess.Popen(
            [str(mat.venv_python), _WORKER_PATH, mat.address.module,
             mat.address.entry],
            cwd=str(mat.repo_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1)
        # Init handshake — send config, receive the interface.
        reply = self._rpc({'cmd': 'init', 'config': config or {}})
        interface = reply.get('interface') or {}
        self._interface_inputs = dict(interface.get('inputs') or {})
        self._interface_outputs = dict(interface.get('outputs') or {})

    # -- RPC ---------------------------------------------------------------
    def _rpc(self, request: dict) -> dict:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise GitProtocolError(
                f'git: worker for {self._materialized.address} is not running '
                f'(exited {proc.returncode if proc else "n/a"}).')
        try:
            proc.stdin.write(json.dumps(request) + '\n')
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise GitProtocolError(
                f'git: worker communication failed: {error}') from error
        if not line:
            stderr = proc.stderr.read() if proc.stderr else ''
            raise GitProtocolError(
                f'git: worker for {self._materialized.address} closed the '
                f'connection unexpectedly. stderr:\n{stderr.strip()}')
        reply = json.loads(line)
        if not reply.get('ok', False):
            raise GitProtocolError(
                f'git: worker error for {self._materialized.address}:\n'
                f'{reply.get("error", "unknown error")}')
        return reply

    # -- Edge interface ----------------------------------------------------
    def inputs(self):
        return self._interface_inputs

    def outputs(self):
        return self._interface_outputs

    def update(self, state, interval):
        reply = self._rpc({
            'cmd': 'update',
            'state': state,
            'interval': interval})
        return reply.get('update') or {}

    def end(self):
        if getattr(self, '_ended', True):
            return
        self._ended = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write(json.dumps({'cmd': 'end'}) + '\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    def __del__(self):
        try:
            self.end()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Conformance (D5, contract 2)
# ---------------------------------------------------------------------------
def _face_of(edge_or_interface) -> Dict[str, dict]:
    if isinstance(edge_or_interface, dict) and (
            'inputs' in edge_or_interface or 'outputs' in edge_or_interface):
        return {
            'inputs': dict(edge_or_interface.get('inputs') or {}),
            'outputs': dict(edge_or_interface.get('outputs') or {}),
        }
    interface = getattr(edge_or_interface, 'interface', None)
    if callable(interface):
        face = interface() or {}
        return {
            'inputs': dict(face.get('inputs') or {}),
            'outputs': dict(face.get('outputs') or {}),
        }
    raise ConformanceError(
        f'cannot read a face from {edge_or_interface!r}')


def conforms(core, resolved, declared_face) -> tuple:
    """Return ``(ok, reason)``: does ``resolved`` (an edge or an interface
    dict) admit every port ``declared_face`` requires, at a resolvable type?
    Over-providing is fine; under-providing / a type mismatch is not."""
    provided = _face_of(resolved)
    required = _face_of(declared_face)
    for direction in ('inputs', 'outputs'):
        need = required.get(direction) or {}
        have = provided.get(direction) or {}
        for port, port_schema in need.items():
            if port not in have:
                return False, (
                    f'{direction[:-1]} port {port!r} required by the declared '
                    f'face is missing from the resolved interface '
                    f'(has {sorted(have)})')
            if core is not None:
                try:
                    core.resolve(port_schema, have[port])
                except Exception as error:
                    return False, (
                        f'{direction[:-1]} port {port!r} does not resolve: '
                        f'declared {port_schema!r} vs resolved '
                        f'{have[port]!r} ({error})')
            elif port_schema != have[port]:
                return False, (
                    f'{direction[:-1]} port {port!r} type mismatch: declared '
                    f'{port_schema!r} vs resolved {have[port]!r}')
    return True, None


def check_conformance(core, resolved, declared_face) -> None:
    """Raise :class:`ConformanceError` naming the mismatch if ``resolved`` does
    not admit ``declared_face``. The authoritative pre-run check (D5) — a run
    never starts against a non-conforming address."""
    ok, reason = conforms(core, resolved, declared_face)
    if not ok:
        raise ConformanceError(f'address does not conform to declared face: {reason}')


# ---------------------------------------------------------------------------
# Protocol type + dispatch
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class GitProtocol(Protocol):
    data: String = field(default_factory=String)


def _bind_class(mat: Materialized):
    """Build a GitRemoteProcess subclass bound to one materialized address
    (mirrors the ray protocol's per-address shadow subclass)."""
    return type(
        f'GitRemote_{mat.address.owner}_{mat.address.repo}',
        (GitRemoteProcess,),
        {'_materialized': mat, '__module__': __name__})


@load_protocol.dispatch
def load_protocol(core, protocol: GitProtocol, data):
    address = parse_git_address(data)
    check_allow_list(address)
    mat = materialize(address)
    bound = _bind_class(mat)

    def instantiate(config, core=None):
        return bound(config, core)

    instantiate.config_schema = bound.config_schema
    # Expose the resolution so callers/tests can record the SHA and do
    # conformance checks (D5) before a run.
    instantiate.materialized = mat
    instantiate.address = address
    instantiate.pin = mat.pin
    return instantiate


def register_types(core):
    core.register_types({'git': GitProtocol})
    return core
