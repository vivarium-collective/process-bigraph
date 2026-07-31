"""
End-to-end tests for the ``git:`` / remote protocol.

These drive the full path against a **lightweight local fixture repo** (a real
git repo with a trivial ``module:callable`` returning a minimal, dependency-free
process). vEcoli's scientific stack is deliberately never pulled in — the
fixture proves the machinery (allow-list, resolve/pin/SHA, per-SHA venv
materialization, stdio-RPC edge, conformance) without the weight.
"""

import os
import json
import shutil
import textwrap
import subprocess

import pytest

from process_bigraph import allocate_core
from process_bigraph.protocols import git as gitproto
from process_bigraph.protocols.git import (
    GitAddress,
    GitProtocolError,
    AllowListError,
    ConformanceError,
    parse_git_address,
    check_conformance,
    conforms,
)


FIXTURE_MODULE = textwrap.dedent('''
    """A dependency-free fixture process for the git: protocol tests."""


    class IncrementProcess:
        def __init__(self, config=None):
            self.rate = float((config or {}).get("rate", 1.0))

        def inputs(self):
            return {"value": "float"}

        def outputs(self):
            return {"value": "float"}

        def interface(self):
            return {"inputs": self.inputs(), "outputs": self.outputs()}

        def update(self, state, interval):
            return {"value": self.rate * float(interval or 1.0)}


    class MismatchProcess(IncrementProcess):
        # Exposes a different face — used to prove conformance rejection.
        def inputs(self):
            return {"temperature": "float"}

        def outputs(self):
            return {"temperature": "float"}


    def make_process(config=None):
        return IncrementProcess(config)


    def make_mismatch(config=None):
        return MismatchProcess(config)
''')

FIXTURE_PYPROJECT = textwrap.dedent('''
    [build-system]
    requires = ["setuptools>=61"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "pbgfixture"
    version = "0.0.1"

    [tool.setuptools]
    py-modules = ["pbgfixture"]
''')

SLUG = 'pbgtest/fixture'


def _git(args, cwd):
    # Inject an identity on every call: some commits happen in a *fetched*
    # checkout (not the fixture repo), which carries no user.name/user.email —
    # locally git falls back to the dev's global config, but CI has none, so a
    # bare `git commit` there fails with exit 128 ("please tell me who you are").
    subprocess.run(
        ['git', '-c', 'user.email=test@example.com', '-c', 'user.name=pbg test',
         *args], cwd=cwd, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture(scope='session')
def fixture_repo(tmp_path_factory):
    """Create a real local git repo exposing ``pbgfixture:make_process``.
    Returns ``(url, default_branch, sha)``."""
    repo = tmp_path_factory.mktemp('pbgfixture-repo')
    (repo / 'pbgfixture.py').write_text(FIXTURE_MODULE)
    (repo / 'pyproject.toml').write_text(FIXTURE_PYPROJECT)
    _git(['init', '--quiet'], repo)
    _git(['config', 'user.email', 'test@example.com'], repo)
    _git(['config', 'user.name', 'pbg test'], repo)
    _git(['add', '.'], repo)
    _git(['commit', '--quiet', '-m', 'fixture'], repo)
    branch = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE).stdout.strip()
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE).stdout.strip()
    return f'file://{repo}', branch, sha


@pytest.fixture
def clean_registry():
    """Isolate allow-list / url-override / cache state per test."""
    gitproto.reset_allow_list()
    yield
    gitproto.reset_allow_list()
    gitproto.clear_repo_url(SLUG)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('PBG_GIT_CACHE', str(tmp_path / 'git_cache'))
    return tmp_path / 'git_cache'


# ---------------------------------------------------------------------------
# Address parsing
# ---------------------------------------------------------------------------
def test_parse_full_address():
    addr = parse_git_address('CovertLab/vEcoli@main#vecoli.workflow:make_process')
    assert addr == GitAddress(
        owner='CovertLab', repo='vEcoli', ref='main',
        module='vecoli.workflow', entry='make_process')
    assert addr.repo_slug == 'CovertLab/vEcoli'


def test_parse_default_ref():
    addr = parse_git_address('CovertLab/vEcoli#vecoli.workflow:make_process')
    assert addr.ref == 'HEAD'


def test_parse_rejects_missing_entry():
    with pytest.raises(GitProtocolError):
        parse_git_address('CovertLab/vEcoli@main')


def test_parse_rejects_garbage():
    with pytest.raises(GitProtocolError):
        parse_git_address('not-an-address')


# ---------------------------------------------------------------------------
# Allow-list (contract 4, D2)
# ---------------------------------------------------------------------------
def test_default_allow_list_has_vecoli(clean_registry):
    assert 'CovertLab/vEcoli' in gitproto.get_allow_list()


def test_un_allow_listed_address_is_refused(clean_registry, cache_dir):
    core = allocate_core()
    protocol = core.access('git')
    with pytest.raises(AllowListError) as excinfo:
        gitproto.load_protocol(
            core, protocol, 'pbgtest/fixture#pbgfixture:make_process')
    assert 'allow-list' in str(excinfo.value)


def test_add_allowed_repo_permits(clean_registry):
    gitproto.add_allowed_repo(SLUG)
    assert SLUG in gitproto.get_allow_list()


# ---------------------------------------------------------------------------
# Resolve + pin + SHA record (contract 1, D3)
# ---------------------------------------------------------------------------
def test_resolve_and_pin_records_sha(clean_registry, cache_dir, fixture_repo):
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)

    addr = parse_git_address(f'{SLUG}@{branch}#pbgfixture:make_process')
    pin = gitproto.resolve_and_pin(addr)

    assert pin.sha == sha
    assert pin.repo_slug == SLUG
    # Pin persisted to disk under the content-addressed cache.
    pin_file = cache_dir / 'pbgtest__fixture' / sha / 'pin.json'
    assert pin_file.exists()
    on_disk = json.loads(pin_file.read_text())
    assert on_disk['sha'] == sha


def test_resolve_default_ref(clean_registry, cache_dir, fixture_repo):
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)
    # No @ref -> resolves the remote's default branch (HEAD).
    addr = parse_git_address(f'{SLUG}#pbgfixture:make_process')
    pin = gitproto.resolve_and_pin(addr)
    assert pin.sha == sha


def test_resolve_refuses_un_allow_listed(clean_registry, cache_dir, fixture_repo):
    url, branch, _ = fixture_repo
    gitproto.set_repo_url(SLUG, url)  # url set, but NOT allow-listed
    addr = parse_git_address(f'{SLUG}@{branch}#pbgfixture:make_process')
    with pytest.raises(AllowListError):
        gitproto.resolve_and_pin(addr)


# ---------------------------------------------------------------------------
# Full materialization + RPC edge (contracts 1 + 3, D1)
# ---------------------------------------------------------------------------
@pytest.fixture
def resolved_edge_factory(clean_registry, cache_dir, fixture_repo):
    """Materialize the fixture once; hand tests a factory that instantiates
    a bound GitRemoteProcess for a given entrypoint + config."""
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)
    core = allocate_core()
    protocol = core.access('git')

    made = []

    def factory(entry='make_process', config=None):
        instantiate = gitproto.load_protocol(
            core, protocol, f'{SLUG}@{branch}#pbgfixture:{entry}')
        edge = instantiate(config or {}, core)
        made.append(edge)
        return instantiate, edge

    yield core, sha, factory

    for edge in made:
        try:
            edge.end()
        except Exception:
            pass


def test_materialize_builds_venv_and_records_sha(resolved_edge_factory, cache_dir):
    core, sha, factory = resolved_edge_factory
    instantiate, edge = factory()
    # SHA recorded on the resolution (reproducibility unit).
    assert instantiate.pin.sha == sha
    # The per-SHA venv actually exists on disk with a python.
    venv_python = cache_dir / 'pbgtest__fixture' / sha / 'venv' / 'bin' / 'python'
    assert venv_python.exists()
    repo_dir = cache_dir / 'pbgtest__fixture' / sha / 'repo'
    assert (repo_dir / 'pbgfixture.py').exists()


def test_rpc_edge_interface(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    _, edge = factory()
    assert edge.inputs() == {'value': 'float'}
    assert edge.outputs() == {'value': 'float'}
    assert edge.interface() == {
        'inputs': {'value': 'float'},
        'outputs': {'value': 'float'}}


def test_rpc_edge_update(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    _, edge = factory(config={'rate': 2.5})
    result = edge.update({'value': 0.0}, 4.0)
    assert result == {'value': 10.0}
    # Second call reuses the live subprocess.
    result2 = edge.update({'value': 0.0}, 2.0)
    assert result2 == {'value': 5.0}


def test_bad_entrypoint_degrades_to_clear_error(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    with pytest.raises(GitProtocolError) as excinfo:
        factory(entry='does_not_exist')
    assert 'does_not_exist' in str(excinfo.value)


def test_isolation_worker_runs_in_repo_venv(resolved_edge_factory, cache_dir):
    """The foreign code runs in the per-SHA venv python, never the host."""
    core, sha, factory = resolved_edge_factory
    _, edge = factory()
    venv_python = cache_dir / 'pbgtest__fixture' / sha / 'venv' / 'bin' / 'python'
    assert str(edge._proc.args[0]) == str(venv_python)


# ---------------------------------------------------------------------------
# Conformance (contract 2, D5)
# ---------------------------------------------------------------------------
def test_conformance_pass(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    _, edge = factory()
    declared = {'inputs': {'value': 'float'}, 'outputs': {'value': 'float'}}
    # Should not raise.
    check_conformance(core, edge, declared)
    ok, reason = conforms(core, edge, declared)
    assert ok and reason is None


def test_conformance_over_provide_ok(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    _, edge = factory()
    # Declared face requires only outputs.value — under-declaring is fine.
    declared = {'outputs': {'value': 'float'}}
    check_conformance(core, edge, declared)


def test_conformance_missing_port_rejected(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    # The mismatch entry provides temperature, not value.
    _, edge = factory(entry='make_mismatch')
    declared = {'inputs': {'value': 'float'}, 'outputs': {'value': 'float'}}
    with pytest.raises(ConformanceError) as excinfo:
        check_conformance(core, edge, declared)
    msg = str(excinfo.value)
    assert "'value'" in msg and 'missing' in msg


def test_conformance_names_mismatch_direction(resolved_edge_factory):
    core, sha, factory = resolved_edge_factory
    _, edge = factory(entry='make_mismatch')
    declared = {'inputs': {'value': 'float'}}
    ok, reason = conforms(core, edge, declared)
    assert not ok
    assert 'input port' in reason and "'value'" in reason


# ---------------------------------------------------------------------------
# Environment provenance (D3) — a warm cache must be verifiable, not assumed
# ---------------------------------------------------------------------------

def test_materialize_reports_how_the_venv_was_built(
        clean_registry, cache_dir, fixture_repo):
    """`install_mode` is part of the result.

    The fixture ships no `uv.lock`, so it must come back RESOLVED. A caller
    that claims reproducibility has to be able to tell that apart from a
    lockfile-frozen build, and the filesystem alone cannot say.
    """
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)
    addr = parse_git_address(f'{SLUG}@{branch}#pbgfixture:make_process')

    materialized = gitproto.materialize(addr)
    assert materialized.install_mode == gitproto.RESOLVED

    # And it survives the warm-cache path, which is the case that matters:
    # a second run must not forget how the first one built the environment.
    assert gitproto.materialize(addr).install_mode == gitproto.RESOLVED


def test_pin_records_the_full_sha(clean_registry, cache_dir, fixture_repo):
    """An abbreviated address still yields an unambiguous recorded commit."""
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)
    addr = parse_git_address(f'{SLUG}@{sha[:8]}#pbgfixture:make_process')

    materialized = gitproto.materialize(addr)
    assert materialized.pin.resolved_sha == sha
    # ...and it is persisted, not just returned.
    pin_file = cache_dir / 'pbgtest__fixture' / sha[:8] / 'pin.json'
    assert json.loads(pin_file.read_text())['resolved_sha'] == sha


def test_checkout_at_the_wrong_commit_is_refused(
        clean_registry, cache_dir, fixture_repo):
    """A cached checkout is verified against the pin, not trusted.

    An interrupted clone or a stray `git checkout` in the cache leaves a
    `.git` behind at the wrong revision. Running it would silently execute
    code that is not the pinned commit — the one thing a SHA is supposed to
    rule out.
    """
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)
    addr = parse_git_address(f'{SLUG}@{branch}#pbgfixture:make_process')
    gitproto.materialize(addr)

    # Add a commit in the cached checkout so HEAD no longer matches the pin.
    repo_dir = cache_dir / 'pbgtest__fixture' / sha / 'repo'
    (repo_dir / 'intruder.txt').write_text('not the pinned commit')
    _git(['add', '.'], repo_dir)
    _git(['commit', '--quiet', '-m', 'drift'], repo_dir)

    with pytest.raises(GitProtocolError) as excinfo:
        gitproto.materialize(addr)
    assert 'not the pinned' in str(excinfo.value)


def test_legacy_ok_stamp_still_loads(clean_registry, cache_dir, fixture_repo):
    """Venvs stamped before provenance existed read back as RESOLVED.

    They were built by whatever path was current then, so `resolved` is the
    honest answer — the conservative one, never a false `locked`.
    """
    url, branch, sha = fixture_repo
    gitproto.add_allowed_repo(SLUG)
    gitproto.set_repo_url(SLUG, url)
    addr = parse_git_address(f'{SLUG}@{branch}#pbgfixture:make_process')
    gitproto.materialize(addr)

    stamp = cache_dir / 'pbgtest__fixture' / sha / 'venv' / '.pbg-installed'
    stamp.write_text('ok')  # the pre-provenance format
    assert gitproto.materialize(addr).install_mode == gitproto.RESOLVED
