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
- **Version:** `0.8.14`.
- **Hooks & UX depth (Claude-Code parity, group 4/4):**
  - **More hook events** — `_run_hooks` now also fires `UserPromptSubmit`, `SessionStart`,
    `SessionEnd`, `PreCompact`, `SubagentStop`, `Notification`. `UserPromptSubmit` can veto a
    prompt (non-zero exit) and its stdout (plus `SessionStart`'s) is injected as context via
    `_HOOK_CONTEXT`/`_inject_context`. `Notification` fires at interactive approval gates.
  - **Output styles** — `/output-style default|concise|explanatory|learning` (or a custom
    `~/.config/hera/output-styles/<name>.md`); appended to the system prompt. `HERA_OUTPUT_STYLE`.
  - **Thinking budget** — `/think off|normal|hard` → `enable_thinking` chat-template kwarg on
    every request (`_think_payload`). `HERA_THINK`.
  - **Status line** — `HERA_STATUSLINE`/config `statusline`: a command fed session JSON whose
    stdout prints above the prompt each turn (`_render_statusline`). `/statusline` shows it.
- **Editing & file tools (Claude-Code parity, group 3/4):**
  - **`multi_edit`** — several exact-string replacements to one file atomically (all-or-nothing;
    `tool_multi_edit`). New tool + schema; in `_EDIT_TOOLS`/`SIDE_EFFECTS` so approval, undo,
    rewind, backstop and verify all cover it. Combined diff preview.
  - **PDF reading** — `read_file` on a `.pdf` extracts text via `pdftotext` (poppler) or a
    pure-Python `pypdf`/`PyPDF2` fallback (`_read_pdf`); helpful install hint if neither is present.
  - **Jupyter notebooks** — `read_file` on `.ipynb` renders cells + outputs (`_render_notebook`);
    new **`notebook_edit`** tool (replace/insert/delete a cell; clears stale outputs on replace).
- **Memory system (Claude-Code parity, group 2/4):**
  - **Hierarchy** — `_memory_sources()` loads enterprise (`/etc/hera/HERA.md` or
    `HERA_ENTERPRISE_MEMORY`) → user (`~/.config/hera/HERA.md`) → the project tree (filesystem
    root down to cwd, first of `HERA.md`/`AGENTS.md`/`AGENT.md` per level). Assembled into the
    system prompt scope-labelled, most-specific last (so cwd overrides). Per-file/total caps.
  - **`@imports`** — `_expand_memory_imports` inlines `@path` lines recursively (depth- and
    cycle-guarded), like CLAUDE.md imports.
  - **`#` quick-add** — typing `# <fact>` appends to project `./HERA.md`; `# user <fact>` →
    user memory (`add_memory`). **`/memory`** lists the loaded hierarchy.
- **Harness robustness (Claude-Code parity, group 1/4):**
  - **Text-format tool-call fallback** — if the server returns no structured
    `tool_calls` but leaks them into the text (Hermes/Qwen `<tool_call>{…}</tool_call>` or
    a fenced JSON object), `_extract_text_tool_calls` recovers them so the agent still acts.
    Balanced-brace scanner (`_scan_json_objects`, string/escape aware); only objects naming a
    *registered* tool are accepted, so illustrative JSON in prose isn't hijacked. Wired into
    `stream_turn` and `_serve_stream`.
  - **Conversation + code `/rewind`** — every user turn records `(msg_index, checkpoint_index)`
    in `TURN_MARKS`; `/rewind [n]` (or a picker) truncates history AND reverts the file edits
    made after that point (`rewind_to`). Marks reset on `/new`·`/clear`·`/logout`·resume.
  - **`/context`** — Claude-Code-style window breakdown (`context_report`): system+project,
    conversation, to-dos, % of `CONTEXT_TOKENS`, and the auto-compaction threshold.
- **Apply-your-edit backstop** — fixes "analyzed the cause but never edited the code." A weak/local
  model often *narrates* a patch instead of emitting an `edit_file` call; the agent loop used to treat
  any tool-call-free turn as "done," so nothing reached disk. Now `run_agent`/`_serve_run` track
  `edit_attempted` and, when the user asked for a change (`_wants_code_change`) but no edit/write tool
  was ever called, inject the `_EDIT_NUDGE` once ("apply it now with edit_file/write_file") and loop.
  Runs before the verify nudge (apply → then verify). Same `AUTO_VERIFY` / `HERA_NO_VERIFY` switch.
- **Verify-your-work loop** — Hera runs/tests the code it writes and fixes failures (Claude-Code/
  Codex style). System prompt directive (primary) + a backstop: `run_agent`/`_serve_run` track
  `edited_code`/`ran_command`; if code was edited but nothing ran, inject the verify nudge once to
  run+fix it. Verify runs/fixes use the normal approval gate. `AUTO_VERIFY` / `HERA_NO_VERIFY`.
  - **Toolchain detection** (`_detect_project_commands`) — prefers `pytest` / `npm test`+`run build` /
    `go build`+`go test` / `cargo` / `make <target>` / Maven·Gradle·Django, fed into the prompt and
    nudge via `_project_hint()`.
  - **Run code it didn't write** (`_wants_run_verification`) — "run the project", "run the tests",
    "make sure it works" etc. trigger the same verify loop with zero edits (`verify_requested`).
- **Claude-Code plan-mode approval flow** — new `exit_plan_mode(plan)` tool the model calls when its
  plan is ready; `_confirm_plan` shows it and asks [1] yes / [2] yes + auto-accept edits / [3] keep
  planning (pluggable `_PLAN_APPROVER`; terminal prompt or serve `plan_review`/`plan_decision` →
  VS Code "Ready to code?" buttons). Approval flips `PLAN_MODE` off and implements. System prompt
  steers the model to the tool.
- **Fixed recurring `400 "System message must be at the beginning"`** — the shared-skills injection
  used to add a SECOND `system` message, which the Qwen template rejects; only fired when a skill
  triggered. Now the skill block is **merged into the single leading system message** (server-side
  in `shared-skills/runtime.py`, shared by the proxy and OWUI). Not a CLI-code change, but the
  symptom users saw. (Distinct from the context-overflow 400, which already self-heals.)
- **`/resume` scoped to the current project** — `_same_project`/`project_sessions`; `/resume` and
  `/sessions` show only this project's conversations (`--list-sessions` still global, `--resume <id>`
  any, `--continue` = latest in this project).
- **Resume by first message, not ID** — sessions now store a `title` (first real user question,
  captured in `save_session` before any compaction; `_first_user` skips `[Summary…]`). `/resume`,
  `/sessions`, `_switch_to` and the startup resume line all show `_session_label(s)` (title +
  date + project folder) and pick by number — no UUID shown. Auto mode confirmed per-project
  (keyed by abs cwd; `/tmp`→all vs `/tmp/projB`→edit are independent).
- **Personal identity + logout** — proxy `/whoami` now also returns `name`; the CLI resolves and
  shows **name + email** (banner `account` row, `hera doctor`, new `hera whoami`, serve `ready.user`
  → VS Code 👤 meta). `logout()` clears key + identity (keeps endpoint): `hera logout`, `/logout`
  (re-onboards inline), serve `{"type":"logout"}`. Globals `USER_EMAIL`/`USER_NAME`/`whoami_label()`.
- **Per-project auto-approve modes** — `AUTO_MODE` ∈ {`read`,`edit`,`all`} (Claude-Code-style),
  saved under `config["auto_modes"][<project path>]`. `/auto read|edit|all|off`, `HERA_AUTO_MODE`,
  serve `{"type":"auto"}`, and a VS Code **Auto** dropdown. Wired into `approve()` **and**
  `_serve_approve()` (the serve gate now also runs plan-mode / permissions / PreToolUse hooks, which
  it previously skipped). Deny rules, plan mode, and hooks still override `all`.
- **`hera doctor`** — `_self_update()` re-downloads `hera.py` over the running file from the
  configured `download_url` (atomic, version-guarded, `--force` to re-pull), then prints a health
  check. The one-word self-updater.
- **Context-overflow self-heal** — `stream_turn`/`_serve_stream` catch a context-size 400
  (`_is_context_overflow`), `compact_history()`, and retry once; `_maybe_auto_compact` also fires on
  the server's real last prompt-token count (`_LAST_PROMPT_TOKENS`). No more raw
  `400 … exceeds the available context size`.
- **Claude-Code parity pass (0.7.0 → 0.8.9):**
  - **To-do tracking** — `todo_write` tool maintains a live checklist (CLI render + `todos`
    serve event → "Plan" block in VS Code). The system prompt nudges it for multi-step tasks.
  - **End-of-task next-step tips** — `_generate_suggestions` (called with `enable_thinking:false`)
    prints 2–3 next steps in the CLI and emits a `suggestions` event (clickable chips in VS Code).
  - **Plan mode** — `/plan`, `HERA_PLAN`, or serve `{"type":"plan"}`; blocks mutating tools even
    under YOLO until the user approves.
  - **Hooks** — `config.json` `hooks` (PreToolUse/PostToolUse/Stop); PreToolUse vetoes on non-zero.
  - **Fine-grained permissions** — `config.json` `permissions` (`allow`/`ask`/`deny`, e.g.
    `run_bash(git *)`, `edit_file(src/**)`); `deny`/`ask` override YOLO.
  - **Auto-compaction** — summarizes history near the context window
    (`HERA_CONTEXT_TOKENS`/`HERA_AUTO_COMPACT_AT`).
  - **Cost** — `HERA_PRICE_IN/OUT` → estimated `$` in turn/session summaries, `/tokens`, serve
    `turn_end`.
  - **Custom slash commands** — `~/.config/hera/commands/*.md` (`$ARGUMENTS`).
  - **Named sub-agents** — `~/.config/hera/agents/*.md` (optional `tools:` frontmatter), via the
    `task` tool's `agent` field.
  - **Background shell** — `run_bash(run_in_background=true)` + `bash_output`/`bash_kill`.
  - **MCP over HTTP/SSE** — `McpHttpClient` (Streamable HTTP) alongside the stdio client; `mcp.json`
    `url` entries with `token`/`headers` (bearer/OAuth token, `${ENV}` expansion).
- **Shared skills:** `/skills` and `/skills <id>`, backed by the proxy's authenticated
  `GET /skills` endpoint; skills live in `shared-skills/skills/*.md` and are shared across the CLI,
  VS Code extension, and Open WebUI web chat.
- **Fast current-time path:** remote `world-time` skill (`GET /skill-api/world-time?q=...`).

## Architecture (all in `hera.py`)
- **Config** — `HERA_*` env (`_env`/`_truthy`). `API_URL`/`API_KEY` have no defaults.
- **Tools** — `TOOLS` dict + `TOOL_SCHEMAS` (OpenAI function schemas): `list_dir`, `read_file`,
  `glob`, `search`, `symbols` (AST/regex code index), `semantic_search` (embeddings; registered
  only if an embeddings endpoint responds), `web_search`/`web_fetch` (keyless DuckDuckGo; on
  unless `HERA_NO_WEB=1`), `write_file`, `edit_file`, `run_bash` (+ `run_in_background`),
  `bash_output`/`bash_kill`, `todo_write`, `install_tool` (on unless `HERA_NO_AUTOINSTALL=1`),
  `task` (optional named `agent`).
- **Web access** — `web_search`/`web_fetch` use `requests` directly (NOT the sandbox), so they
  reach the network even though `run_bash` is network-isolated. Read-only → no approval gate.
- **Installing tools** — proactive `install_tool` (in `SIDE_EFFECTS`, user approves) and reactive
  `run_bash` exit-127 detection → `_offer_install` → retry once. Both share `_do_install()`, which
  runs the package manager **unsandboxed** (`_install_plan`: apt/dnf/pacman/brew/apk, `sudo -n`).
  Consent is pluggable via `_INSTALL_APPROVER` (terminal `input()` in the REPL; an IDE
  `approval_request` event in `--serve`).
- **Approval gate** — `approve()` + `bash_allowed()`; `SIDE_EFFECTS` need approval; allowlist
  (`HERA_ALLOW`/`.heraallow`/`/allow`) + built-in denylist. Layered on top: **plan mode** (blocks
  mutating tools), **config `permissions`** (`_perm_decision`: allow/ask/deny), and **PreToolUse
  hooks** (`_run_hooks` can veto) — all consulted before the YOLO/allowlist short-circuit.
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
- **Extensions** — MCP stdio client (`McpClient`) **and HTTP/SSE client (`McpHttpClient`,
  Streamable HTTP, bearer/OAuth token)** from `~/.config/hera/mcp.json` (an entry with `command`
  is stdio; with `url` is HTTP), plus custom tools (`~/.config/hera/tools.py`, a `HERA_TOOLS`
  list). `register_extensions()`.
- **Custom commands / agents** — `~/.config/hera/commands/*.md` (slash commands, `$ARGUMENTS`) and
  `~/.config/hera/agents/*.md` (named sub-agents) loaded via `_load_markdown_dir`.
- **Headless mode** — `--serve`: newline-delimited JSON on stdin/stdout (`ready`/`reasoning`/
  `token`/`tool_start`/`approval_request`/`tool_end`/`turn_end`/`todos`/`suggestions`), used by
  the VS Code extension. Inputs include `{"type":"plan"}` to toggle plan mode.
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
