"""Keep pytest from collecting fixture packages as test modules.

pytest.ini sets ``python_files = *.py`` (broader than the usual
``test_*.py``), so any ``.py`` file under this tree is a collection
candidate by default — including ``fixtures/fake_generator_pkg/`` files
like ``composites.py``, whose ``@composite_generator`` decorator has a
real side effect (registering into the shared process_bigraph.composite_spec
registry). If pytest's own collection import fires that decorator, the
resulting registration can outlive collection in ``sys.modules`` — so a
LATER test's explicit ``importlib.import_module("fake_generator_pkg")``
(e.g. in ``discover_generators()``) becomes a no-op cache hit that never
re-fires the decorator, while an unrelated test's registry-clearing
fixture can still wipe the collection-time registration first. The net
effect is a full-suite-only flaky failure that never reproduces when the
affected test file is run in isolation. Excluding `fixtures/` from
collection removes the root cause: fixture packages are data for tests
to install/import explicitly, never test modules themselves.
"""
collect_ignore = ["fixtures"]
