# Contributing to Hera CLI

Thanks for your interest in improving Hera! This is a small, single-file project,
so contributing is straightforward once you know the layout and a couple of
project-specific quirks.

## Heads-up: this repo is mirrored from upstream

The README notes that this repository is **kept in sync from the "gpustack" stack**
(via a `scripts/sync-hera-cli.sh` that lives there, not here). In practice that
means:

- Commits on `main` here are periodic sync dumps from the upstream source of truth.
- A change merged **only** here can be overwritten by the next sync.

So when you open a PR, please mention in the description that the fix likely needs
to land **upstream** too. The maintainer can fold it back into the gpustack stack so
it sticks. Don't let this stop you from contributing here — a clear PR makes the
upstream change easy.

## Project layout

Everything lives in one file: **`hera.py`** (~6,900 lines). There is no framework
and no package structure — that's intentional, so a single download is the whole
tool.

| Path | What it is |
|------|------------|
| `hera.py` | The entire CLI: config, tools, the agent loop, the REPL, `main()`. |
| `install.sh` | The one-line installer (downloads `hera.py`, adds `requests`). |
| `VERSION` | Single source of the version string. Bump when behavior changes. |
| `README.md` | User-facing overview and install instructions. |
| `ACCESS_*.md` | Access guides for the CLI, VS Code extension, and web. |
| `docs/` | Supplementary docs (e.g. `SANDBOX_TEST.md`). |

Inside `hera.py`, the rough top-to-bottom order is:

1. **Config & identity** — layered resolution: env var → `~/.config/hera/config.json` → default (`_cfg`).
2. **Project/git context detection** — sniff the project's run/test commands.
3. **Sandboxing** — `bubblewrap` / `unshare` wrappers for `run_bash`.
4. **Terminal rendering** — ANSI colors, the `SYM_*` glyph constants, markdown streaming.
5. **Tools** — the `tool_*` functions (read/write/edit files, search, run_bash, web, …).
6. **Tool registry** — `TOOLS`, `TOOL_SCHEMAS`, `SIDE_EFFECTS`.
7. **Model transport** — `stream_turn()` (OpenAI-compatible and Anthropic paths).
8. **The agent loop** — `run_agent()`, the reason→act loop with self-verify nudges.
9. **Sessions/extras** — checkpoints/undo, sub-agents, todos, MCP.
10. **`main()`** — arg parsing, `--serve` mode, and the interactive REPL.

## Requirements

- **Python 3.7+**
- **`requests`** — the only third-party dependency.
- Optional, for full functionality:
  - `bubblewrap` (`sudo apt install bubblewrap`) — filesystem-confined `run_bash`.
  - `pdftotext` (poppler) or the `pypdf` package — reading PDFs.

```bash
pip install requests
```

## Development setup

```bash
# work against your fork
git clone git@github.com:<you>/hera-cli.git
cd hera-cli
git checkout -b my-fix

chmod +x hera.py
./hera.py --help        # no endpoint needed for --help
```

To run it against a model you'll need an OpenAI-compatible endpoint and key
(see the README); for most code changes you can verify without a live endpoint
(see below).

## Verifying your change

There is **no automated test suite** yet, so verify deliberately:

1. **It still compiles** — this catches the most common breakage:
   ```bash
   python3 -m py_compile hera.py
   ```
2. **It runs** — `--help` exercises a lot of module-level code with no endpoint:
   ```bash
   ./hera.py --help
   ```
3. **Targeted check** — if you touched a specific function, import and call it,
   or drive the relevant `tool_*` function directly:
   ```bash
   python3 -c "import hera; print(hera.tool_glob('**/*.py', '.'))"
   ```
4. **Live run** (when relevant) — point at your endpoint and exercise the path
   end to end. Prefer this for anything touching the agent loop, streaming, or
   tool approval.

If you add a meaningful, self-contained fix, a tiny `python3 -c` reproduction in
the PR description (before/after) goes a long way.

## Style

Match the surrounding code — it's consistent, so follow what's already there:

- 4-space indent, roughly ≤100 columns.
- `snake_case` functions; tool entry points are named `tool_<name>`.
- Module-private helpers start with `_` (e.g. `_cfg`, `_resolve`).
- Keep it dependency-free. Don't add a third-party import for something the
  standard library can do — `requests` is the only allowed runtime dep.
- The code is flake8-clean; blind `except` blocks are deliberately marked
  `# noqa: BLE001` with a comment on why they can't be allowed to fail.

### Display symbols

User-facing glyphs go through the `SYM_*` constants (defined around line 956),
each with an ASCII fallback, e.g. `SYM_EMDASH = _u("—", "--")`. **Interpolate the
constant** — never write the name as bare text:

```python
print(f"done {SYM_ELLIPSIS}")     # ✅ renders "done …" (or "done ..." on ASCII terminals)
print("done SYM_ELLIPSIS")        # ❌ prints the literal text "done SYM_ELLIPSIS"
```

In docstrings, comments, and `--help` text — which run before the constants are
defined and aren't routed through the terminal-symbol system — just use the
literal glyph (`—`, `→`, `…`).

## Adding a tool

Tools are the model's hands. To add one:

1. Write a `tool_<name>(...)` function that returns a **string** (what the model
   reads back). Return `"[error] ..."` strings rather than raising.
2. Register it in `TOOLS`.
3. Add its JSON schema to `TOOL_SCHEMAS` (name, description, parameters).
4. If it changes the world (writes files, runs commands, hits the network with
   side effects), add its name to `SIDE_EFFECTS` so it requires approval.

Keep descriptions tight and accurate — the model relies on them to choose tools.

## Submitting a pull request

1. Branch from `main` with a descriptive name (`fix/...`, `docs/...`, `feat/...`).
2. Keep the change focused; one concern per PR.
3. Confirm `python3 -m py_compile hera.py` passes and `./hera.py --help` works.
4. Bump `VERSION` if you changed user-visible behavior.
5. In the PR description, explain the problem, the fix, and how you verified it —
   and note whether it should also be applied **upstream** (see the top of this
   file).

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened (include the
exact error text), your OS, and `python3 --version`. A minimal repro is gold.
