# The `git:` / remote protocol — resolving a repo address to a runnable process

**Status:** Design (Layer 3 capability — decisions-forward)
**Date:** 2026-07-31
**Repo:** process-bigraph (a new `Protocol`, alongside `local:`)
**Companion:** `2026-07-31-study-investigation-templates-design.md` (§5.1 — the comparison-harness needs this)
**Why:** an **address site** (Layer-1 §4.6) can be filled with a *repo* address — the vEcoli case: `fill(vecoli.address, 'git:CovertLab/vEcoli@<ref>')`. That address must resolve to a runnable `Edge` whose face is checked. This is the one genuinely new capability behind "compare against a different implementation."

---

## 1. What it does

`git:<owner>/<repo>@<ref>#<entry>` resolves to an `Edge`:

1. **Fetch + pin** the repo at `<ref>` (branch/tag/commit) into a content-addressed cache; record the resolved **commit SHA** (a moving `<ref>` re-resolves; a SHA is frozen — the reproducibility unit).
2. **Materialize an environment** for that checkout (its deps).
3. **Expose `<entry>`** (a designated process/composite entrypoint) as an `Edge` whose `interface()` is the face the address site checks (`admits`).

Filling the site is one call; the *heavy* work (fetch/install) is cached and happens once per SHA, **not** at document build. Conformance is checked **before a run**, not at every fill.

---

## 2. The decisions (each needs a call — recommendation given)

### D1. Environment isolation — *the load-bearing decision*
Running fetched code (vEcoli pulls a large scientific stack) must not pollute or break the host env.
- **(a) Subprocess + per-SHA venv** — `uv venv` per commit SHA, install the repo, run the entrypoint out-of-process, talk over a small RPC/stdio boundary. Portable, no daemon, matches process-bigraph's existing detached-run model. **Recommended.**
- (b) Container (Docker) per repo — strongest isolation + reproducibility, but a hard host dependency (not everywhere) and heavier.
- (c) In-process import — cheapest, but foreign deps collide with the host (numpy pins, etc.) and one bad import can crash the server. **Rejected** for anything but trusted, pip-clean repos.

*Recommendation: (a). The edge is a thin RPC proxy over a subprocess running in the repo's own venv; the `interface()` it advertises is the face.*

### D2. Trust & security — *running foreign code*
`git:` executes code from a URL.
- **Allow-list of orgs/repos** in `workspace.yaml` (default: `CovertLab/vEcoli` + explicitly added forks); a `git:` address outside the list is refused, not run.
- **Pin-to-SHA for runs**: a bare branch is resolved-and-recorded; the *run* uses the recorded SHA, and a changed SHA is surfaced (never silently re-run).
- No credentials in the address; private repos use the host's existing git auth.
- *Recommendation: allow-list + SHA-pinning are mandatory; sandboxing beyond the venv/subprocess is out of scope v1 (documented as "runs code you allow-listed").* **This is a real user decision — confirm the allow-list model.**

### D3. Caching & reproducibility
- Cache key = `(repo, commit-SHA)`; the built venv is cached too (keyed by SHA + a lockfile hash). A pinned SHA is reproducible; a comparison investigation records the SHA in its artifacts.
- GC/eviction of old checkouts is a workspace maintenance concern (out of scope v1; just don't unbounded-grow silently — log sizes).

### D4. The entrypoint contract
How does `git:` know *what* in the repo to run?
- **(a) `#<entry>`** in the address names a module:callable the repo exposes (e.g. `#vecoli.workflow:make_process`). Explicit, no convention needed. **Recommended.**
- (b) A repo-root manifest (`.pbg-entry.yaml`) the protocol reads. Cleaner addresses, but requires the repo to adopt a convention (vEcoli is CovertLab's, not ours).
- *Recommendation: (a), with (b) as an optional convenience later. For vEcoli we point `#` at a thin adapter we maintain in v2ecoli that wraps CovertLab's entrypoint to the study's face.*

### D5. Conformance timing
- `admits` at **fill** time checks the *declared* face if the address is already resolved-and-cached; otherwise it's deferred to **pre-run** (resolve → introspect `interface()` → conform). A run never starts against a non-conforming address.
- *Recommendation: cheap declared-face check at fill when available; authoritative check pre-run.*

---

## 3. Contracts

1. **Resolution is a pure function of `(repo, SHA, entry)`** → the same `Edge` face; recorded SHA is the reproducibility unit.
2. **`admits` holds against the resolved `interface()`** — a non-conforming vEcoli entrypoint is rejected before a run, naming the face mismatch.
3. **Isolation** — foreign deps never enter the host process/env (D1(a)); a fetched-code failure degrades to a clear error, not a host crash.
4. **Allow-list** — an un-allow-listed `git:` address is refused (D2).

---

## 4. Sequencing & out of scope

**Sequence:** (1) the protocol skeleton + cache/pin + allow-list (no real repo); (2) subprocess+venv materialization + RPC edge on a *lightweight* test repo; (3) a v2ecoli-side vEcoli adapter (`#` entry) so the comparison-harness (companion spec §5) runs against `CovertLab/vEcoli@<ref>`; (4) SHA recording in comparison artifacts.

**Out of scope v1:** container isolation (D1(b)); full sandboxing beyond venv/subprocess; private-repo credential management beyond host git auth; cache GC policy.

**Open for the user:** confirm the **allow-list trust model** (D2) and the **subprocess+venv isolation** (D1) before build — these define how the workbench runs foreign code.
