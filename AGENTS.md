# AGENTS.md

Guidance for AI coding agents (Codex, Cursor, Aider, Junie, Gemini CLI, …) working on the
**Akgentic framework** monorepo. This file follows the [agents.md](https://agents.md) open
format and complements `README.md` and `CONTRIBUTING.md`.

> Human contributors: read `README.md` first. This file is a *README for agents*.

---

## Golden rules

1. **Use only currently available packages.** Do not add new third-party dependencies,
   bump major versions, or introduce new tools/services that are not already declared in
   the workspace `pyproject.toml` files (root and `packages/*/pyproject.toml`) or in
   `packages/akgentic-frontend/package.json`. Reuse what already exists.
2. **When something is not possible — ask the user.** If a task cannot be completed with
   the current packages, configuration, or constraints (missing dependency, missing API
   key, ambiguous requirement, blocked by infrastructure, conflicting tests, etc.),
   **stop and ask for feedback** instead of guessing, inventing APIs, installing new
   packages, or weakening tests.
3. **Do not weaken or skip tests** to make a task pass. Never delete, `@disable`,
   `@pytest.mark.skip`, comment out, or stub assertions of existing tests just to get
   green CI. If a test seems wrong, raise it with the user.
4. **Stay inside the workspace.** Don't create files outside the project root unless
   explicitly requested. Don't touch `.junie/` unless the task is specifically about
   project guidelines.
5. **Closest AGENTS.md wins.** Per-package `AGENTS.md` files (if present under
   `packages/<name>/`) override this root file for changes within that package.

---

## Project overview

**Akgentic** is a modern actor-based, multi-agent framework for **Python 3.12+**, plus an
Angular 19 frontend. The repository is a **uv workspace monorepo** with eight packages:

| Package | Role | Depends on |
|---|---|---|
| `akgentic-core` | Actor framework (Pykka), messaging, orchestrator | — |
| `akgentic-llm` | Multi-provider LLM integration, REACT pattern | — |
| `akgentic-tool` | Tool abstractions, workspace, planning, search, MCP | core |
| `akgentic-team` | Team lifecycle, event sourcing, YAML/MongoDB | core |
| `akgentic-agent` | LLM-powered agents, typed message routing | core, llm, tool |
| `akgentic-catalog` | Configuration registry (YAML/MongoDB) | core, llm, tool, team |
| `akgentic-infra` | Backend (community / department / enterprise tiers) | core, llm, tool, agent, catalog, team |
| `akgentic-frontend` | Angular SPA (REST + WebSocket client) | — |

Lower layers must **never import from higher layers**. Respect the dependency graph.

---

## Repository layout

```
.
├── AGENTS.md                  # this file
├── README.md                  # human-facing quick start
├── CONTRIBUTING.md            # contribution guidelines
├── pyproject.toml             # uv workspace root, shared pytest/mypy/coverage config
├── uv.lock                    # locked dependency versions — do not edit by hand
├── packages/
│   ├── akgentic-core/         # each package: src/, tests/, pyproject.toml, README.md
│   ├── akgentic-llm/
│   ├── akgentic-tool/
│   ├── akgentic-team/
│   ├── akgentic-agent/
│   ├── akgentic-catalog/
│   ├── akgentic-infra/
│   └── akgentic-frontend/     # Angular app (npm)
├── src/                       # quick-start example apps
│   ├── infra_server.py        # backend launcher
│   ├── agent_team/            # programmatic team example
│   ├── accounting_team/       # domain example
│   └── extract/               # extraction example
├── scripts/                   # helper shell scripts (sandbox image, sync packages)
├── data/                      # default catalog & event store (gitignored content)
└── workspaces/                # agent workspace artifacts
```

Each package is git-tracked as a submodule-like entry; treat `packages/<name>` as the unit
of ownership for that library.

---

## Setup commands

Prerequisites: **Python ≥ 3.12**, [`uv`](https://docs.astral.sh/uv/),
**Node.js ≥ 20** and **npm** (frontend only). Do not switch package managers.

```bash
# Clone with submodules
git clone https://github.com/b12consulting/akgentic-framework.git
cd akgentic-framework

# Create the shared virtual environment and install all workspace packages
uv venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv sync --all-packages --all-extras
```

Frontend:

```bash
cd packages/akgentic-frontend
npm install
```

> If `uv sync` or `npm install` fails because of a missing tool/version, **ask the user**
> instead of installing alternatives globally.

---

## Run commands

```bash
# Backend server (community tier, FastAPI on :8000)
source .venv/bin/activate
export OPENAI_API_KEY="..."          # required for LLM calls
export TAVILY_API_KEY="..."          # required for SearchTool
python src/infra_server.py

# Web UI on http://localhost:4200
cd packages/akgentic-frontend && npm start

# CLI examples
python src/agent_team/main.py        # programmatic team
python src/catalog/main.py           # YAML catalog-driven team
```

Default state directories: `./data/catalog/` and `./data/event_store/`. They are
configurable via `CommunitySettings` or `AKGENTIC_*` environment variables.

If an API key is missing, **do not** hard-code a fake key — ask the user.

---

## Test & quality commands

The repo uses a **shared** pytest / mypy / coverage configuration (see root
`pyproject.toml`). Run all commands from the repo root inside the activated venv.

```bash
# Run the full Python test suite
uv run pytest

# Run tests for a single package
uv run pytest packages/akgentic-core/tests
uv run pytest packages/akgentic-agent/tests -k "routing"

# Skip integration tests that require live LLM/API keys
uv run pytest -m "not integration"

# Type checking (strict mypy)
uv run mypy packages/akgentic-core/src
uv run mypy packages/akgentic-agent/src

# Coverage (must stay ≥ 80%, see [tool.coverage.report])
uv run pytest --cov

# Frontend
cd packages/akgentic-frontend
npm run lint
npm test
npm run build
```

Rules:

- All new/changed Python code must pass **`mypy --strict`** unless the module is in the
  pre-existing override list in root `pyproject.toml`. Do not extend that override list.
- Tests marked `@pytest.mark.integration` require real API keys; do not run them in
  offline contexts.
- Coverage must stay at or above **80%**. Do not lower `fail_under`.
- Before submitting, run at minimum the affected package's tests and `mypy` on the
  changed `src/` directory.

---

## Code style

### Python (all `akgentic-*` packages)

- **Python 3.12+ only.** Use modern syntax: PEP 695 generics where appropriate,
  `match` statements, `from __future__ import annotations` is **not** required.
- **Type hints everywhere.** Public functions/methods must be fully annotated; mypy is
  strict.
- **Pydantic v2** for all data models and message schemas (`Message`, `AgentMessage`,
  `AgentCard`, `ToolCard`, `TeamCard`, …). Use `BaseModel`, `Field`, `model_validator`.
- **Actor / message conventions:**
  - Subclass `Akgent` (Pykka-based) for actors.
  - Define typed messages as `Message` subclasses; never pass raw dicts between actors.
  - Receive handlers follow the `receiveMsg_<MessageClass>` naming pattern.
  - Respect the typed `AgentMessage` 5-intent protocol (`request`, `response`,
    `notification`, `instruction`, `acknowledgment`) in `akgentic-agent`.
- **Imports:** absolute, grouped stdlib / third-party / first-party (`akgentic.*`).
- **Errors:** prefer `RetriableError` from `akgentic-tool` for recoverable failures so
  the REACT loop can retry.
- **Logging:** use the standard `logging` module; do not `print` in library code.
  `logfire` is opt-in via the `--logfire` server flag.
- **Async:** `pytest` runs in `asyncio_mode = "auto"`. Don't manually wrap with
  `asyncio.run` inside tests.
- **Formatting:** match the surrounding file. Don't reformat unrelated code.

### TypeScript / Angular (`akgentic-frontend`)

- Angular 19 + PrimeNG 19, RxJS, ngx-echarts, ngx-markdown, Monaco Editor.
- TypeScript strict mode; follow the existing ESLint and Prettier configs.
- Don't add new UI component libraries; reuse PrimeNG.

### Markdown / docs

- Keep `README.md` human-facing and concise. Detailed agent-only instructions belong
  here in `AGENTS.md` or in per-package `AGENTS.md` files.
- Use fenced code blocks with language tags.

---

## Adding code — checklist

Before opening a PR, verify:

- [ ] Change lives in the correct package per the dependency graph (no upward imports).
- [ ] No new third-party dependency added. If one is truly needed, **ask the user**.
- [ ] Public API has type hints and (where user-facing) a docstring matching the
      style of nearby code.
- [ ] Tests added/updated next to the package (`packages/<name>/tests/`). New behavior
      has a unit test; bug fixes have a regression test.
- [ ] `uv run pytest` passes for affected packages.
- [ ] `uv run mypy` passes for affected `src/` paths.
- [ ] Coverage still ≥ 80%.
- [ ] No secrets, API keys, or absolute local paths committed.
- [ ] `README.md` / per-package `README.md` updated if user-visible behavior changed.

---

## Security & secrets

- **Never commit** `OPENAI_API_KEY`, `TAVILY_API_KEY`, `ANTHROPIC_API_KEY`, MongoDB
  URIs, OAuth client secrets, or any other credentials. Use environment variables.
- The `.gitignore` excludes `.env`, `data/`, and local workspaces — keep it that way.
- The `WorkspaceTool` gives agents filesystem access. When extending it, do not relax
  path-sandboxing checks.
- The `MCPTool` can launch subprocesses (stdio servers); don't disable command
  whitelists.

---

## Commit & PR conventions

- Conventional, imperative commit subjects: `feat(core): …`, `fix(agent): …`,
  `docs: …`, `test(tool): …`, `refactor(catalog): …`, `chore: …`.
- One logical change per PR; keep diffs small and focused.
- Reference the affected package(s) in the subject scope.
- Update `CHANGELOG`-style notes only if the package already maintains one.
- Do **not** initiate commits autonomously unless the user asks.

---

## When to ask the user

Stop and request feedback whenever you encounter any of these situations — do not
silently work around them:

- A task seems to require a **package, library, service, or model that is not already
  installed/configured** in this repo.
- A test fails after **3 genuine fix attempts** and the root cause is unclear, or the
  test itself looks incorrect.
- An API key, secret, or external endpoint is missing.
- Requirements are ambiguous, contradict existing code, or conflict with the
  dependency graph above.
- A change would touch **multiple packages across layers** in a way that risks
  breaking the architecture.
- You would otherwise need to disable, skip, or weaken existing tests, lower coverage,
  or extend the mypy override list.
- Anything destructive (deleting files you didn't create, dropping data in
  `data/`, force-pushing, rewriting history).

When in doubt: **ask first, code second.**

---

## Working with `docs/tasks.md`

The file `docs/tasks.md` is the project's development checklist. Follow these rules when
updating it:

- **Mark completed tasks** by changing `[ ]` to `[x]` in the Status column.
- **Keep phases intact.** Do not remove, reorder, or renumber existing phases or tasks.
- **Adding tasks is allowed.** Insert new rows at the end of the relevant phase table
  using the next sequential number (e.g. `1.12` after `1.11`). Every new task **must**
  include a `Plan` reference and a `Requirements` reference — no orphan tasks.
- **Modifying tasks.** If you update a task description, ensure its `Plan` and
  `Requirements` links remain accurate and up to date.
- **Formatting.** Preserve the existing Markdown table style: `| # | Task | Plan |
  Requirements | Status |`. Use the same column alignment and link notation
  (`Plan: …`, `Req: …`).
