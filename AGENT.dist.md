# Hera CLI — Agent Notes

Handoff notes for an agent continuing work on the **Hera** agentic coding CLI in this repo.
(Hera also loads this file as project context when launched here, since `AGENT.md` is one of its
context files.)

> This repo is **public and intentionally secret-free**: `hera.py` and `install.sh` hardcode
> **no server host and no API key**. Keep it that way — users supply `HERA_API_URL` +
> `HERA_API_KEY` at runtime. Never commit a host/IP, key, or token here.

## What this is
`hera.py` is a single-file (deps: only `requests`) agentic CLI for an OpenAI-compatible chat
endpoint. It runs the model in a reason→act loop with real tools and a permission/sandbox model,
aiming for Claude-Code-class behavior.

## Latest changes
- **Version:** `0.6.3`.
- **Shared skills:** the CLI now has `/skills` and `/skills <id>`, backed by the proxy's
  authenticated `GET /skills` endpoint. Shared prompt/workflow skills live upstream in
  `shared-skills/skills/*.md`, can trigger automatically or via `@skill:<id>` / `/skill <id>`,
  and are shared across the CLI, VS Code extension, and Open WebUI web chat.
- **Fast current-time path:** upstream added a remote `world-time` skill served from
  `GET /skill-api/world-time?q=...` on the proxy. That keeps exact time/date/day prompts out of
  flaky web-search retrieval and returns concrete live time data instead.

## Architecture (all in `hera.py`)
- **Config** — `HERA_*` env (`_env`/`_truthy`). `API_URL`/`API_KEY` have no defaults.
- **Tools** — `TOOLS` dict + `TOOL_SCHEMAS` (OpenAI function schemas): `list_dir`, `read_file`,
  `glob`, `search`, `symbols` (AST/regex code index), `semantic_search` (embeddings; registered
  only if an embeddings endpoint responds), `web_search`/`web_fetch` (keyless DuckDuckGo; on
  unless `HERA_NO_WEB=1`), `write_file`, `edit_file`, `run_bash`, `install_tool` (on unless
  `HERA_NO_AUTOINSTALL=1`), `task`.
- **Web access** — `web_search`/`web_fetch` use `requests` directly (NOT the sandbox), so they
  reach the network even though `run_bash` is network-isolated. Read-only → no approval gate.
- **Installing tools** — proactive `install_tool` (in `SIDE_EFFECTS`, user approves) and reactive
  `run_bash` exit-127 detection → `_offer_install` → retry once. Both share `_do_install()`, which
  runs the package manager **unsandboxed** (`_install_plan`: apt/dnf/pacman/brew/apk, `sudo -n`).
  Consent is pluggable via `_INSTALL_APPROVER` (terminal `input()` in the REPL; an IDE
  `approval_request` event in `--serve`).
- **Approval gate** — `approve()` + `bash_allowed()`; `SIDE_EFFECTS` need approval; allowlist
  (`HERA_ALLOW`/`.heraallow`/`/allow`) + built-in denylist.
- **Prompt UI** — raw-mode line editor (`RawLineReader`, stdlib `termios`/`tty`) with a live
  `/`-command dropdown (filter, `↑`/`↓`, `Tab`/`Enter`); falls back to `input()` when not a TTY.
  `SLASH_COMMANDS` is the single source of truth for the menu, `/help`, `/resume`, and `/skills`.
- **Sandbox** — `_sandbox_argv()`/`_sandbox_wrap_argv()`: bwrap (fs confined to cwd, net off) →
  unshare → none. `run_bash` and (optionally) MCP servers run under it.
- **Streaming** — `stream_turn()` parses SSE, streams reasoning + content (line-buffered
  markdown render), assembles tool calls. `run_agent()` drives the loop; `_exec_call()` runs one
  tool (shared with sub-agents).
- **Sub-agents** — `task` tool → `run_subagent()` (full toolset minus `task`).
- **Sessions** — saved per user under `~/.config/hera/sessions/<HERA_USER or key-hash>/`;
  `--resume`/`--continue`/`--list-sessions`, `/sessions`, `/resume` (interactive picker → `_switch_to`), `/new`.
- **Checkpoints** — every write/edit snapshots prior state; `/undo` reverts (`CHECKPOINTS`).
- **Extensions** — MCP stdio client (`McpClient`, `~/.config/hera/mcp.json`) and custom tools
  (`~/.config/hera/tools.py`, a `HERA_TOOLS` list). `register_extensions()`.
- **Headless mode** — `--serve`: newline-delimited JSON on stdin/stdout (`ready`/`reasoning`/
  `token`/`tool_start`/`approval_request`/`tool_end`/`turn_end`), used by the VS Code extension.
  **stdout stays JSON-only — tool functions must `print()` progress to `sys.stderr`**, never
  stdout, or they corrupt the protocol. The reactive install offer surfaces as an
  `approval_request` here via `_serve_install_approver` (set on `_INSTALL_APPROVER`).
- **TUI** — theme honors `NO_COLOR`/`HERA_NO_COLOR`/`HERA_FORCE_COLOR`; wordmark banner;
  `render_md_line()` for answers.

## Extending it
- **Add a tool:** write `def tool_x(**kw) -> str`, add to `TOOLS`, append a schema to
  `TOOL_SCHEMAS`; add to `SIDE_EFFECTS` if it mutates. For the `--serve` path it flows through
  automatically.
- **Mirror to `--serve`:** the serve path (`_serve_exec`/`_serve_stream`) reuses `TOOLS` and the
  schemas, so new tools work in the webview too; only display events differ.

## Repo / workflow
- This repo is **generated/synced** from an upstream stack; the canonical source for `hera.py`,
  `install.sh`, `README.md`, the `ACCESS_*.md` guides, and this file lives there. Prefer changing
  it upstream and re-syncing; direct edits here can be overwritten by the next sync.
- Keep it secret-free and host-free (see top). Before committing, scan the tree for any IPv4
  address, `sk-…` API keys, or `ghp…`/`github_pat…` tokens — there should be none.

## Try it
```bash
pip install requests
export HERA_API_URL=http://<host>/v1   # your endpoint
export HERA_API_KEY=<your key>
python3 hera.py
```
See `ACCESS_CLI.md` for the full setup and the per-user account flow.
