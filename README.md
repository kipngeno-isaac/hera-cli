# Hera CLI

**Hera** is a single-file, agentic coding CLI for the Qwen3.6-35B-A3B model (served via an
OpenAI-compatible llama.cpp endpoint). It runs the model in a reason→act loop with real tools:
it lists directories, finds and greps code, reads/writes/edits files, and runs shell commands
**in the directory you launch it from**. It asks for approval before editing files or running
commands, streams the model's reasoning, and tracks token usage.

> Single file, no framework. Requires Python 3.7+ and `requests`.
> This repo is kept in sync from the gpustack stack; see `scripts/sync-hera-cli.sh` there.

## Other ways to access the same model

This repo is the standalone **terminal CLI**. The same model and your same account are also
reachable two other ways (provided by the full stack, not this repo):

- **VS Code / Cursor** — an extension with a chat panel, in-editor diffs, and LSP-driven context
  (it drives this CLI via `hera --serve`).
- **Web** — an Open WebUI browser chat with per-user, email-based login.

Your **Open WebUI API key** is the single credential across all three (terminal, editor, web).
Full guides: [`ACCESS_CLI.md`](ACCESS_CLI.md) · [`ACCESS_VSCODE.md`](ACCESS_VSCODE.md) ·
[`ACCESS_WEB.md`](ACCESS_WEB.md).

---

## Install

**Requirements:** Python 3.7+ and the `requests` library. That's it — `hera.py` is a single
file with no other dependencies.

### Easiest — straight from this public GitHub repo

```bash
pip install requests
curl -fsSL https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py -o ~/.local/bin/hera
chmod +x ~/.local/bin/hera
```

(Or `git clone https://github.com/jones0011738/hera-cli && chmod +x hera-cli/hera.py`.)
Make sure `~/.local/bin` is on your `PATH`.

> Optional: install `bubblewrap` (`sudo apt install bubblewrap`) for full `run_bash` sandboxing.

### Easiest — the server's installer (endpoint is pre-configured)

If your admin runs the download server, the one-liner installer fetches `hera`, adds `requests`,
puts it on your `PATH`, **and saves the endpoint** so you only ever paste your key:

```bash
HERA_SERVER=http://<HOST>:8081 bash <(curl -fsSL http://<HOST>:8081/install.sh)
hera        # first run: paste your Open WebUI API key once — done
```

### Then point it at your server (only if you installed manually)

First run is interactive — Hera asks for the endpoint (the identity proxy, `http://<HOST>:8090/v1`)
and your key, then saves both to `~/.config/hera/config.json` (mode 600) so it never asks again.
To skip the prompt, write the file yourself:

```bash
mkdir -p ~/.config/hera
printf '{ "api_url": "http://<HOST>:8090/v1" }\n' > ~/.config/hera/config.json
hera        # paste your key once
```

Environment variables still work and **override** the file (handy for CI):

```bash
export HERA_API_URL=http://<HOST>:8090/v1   # the identity proxy
export HERA_API_KEY=sk-...                  # your Open WebUI API key
export HERA_USER=you@example.com            # optional: labels your sessions
```

That's why this repo can be public: it ships **no key and no host** — you supply both, once. See
[`ACCESS_CLI.md`](ACCESS_CLI.md) for the full walkthrough.

---

## Run

```bash
cd ~/my-project               # Hera works in the current directory
hera                          # uses your saved config (or env vars)
```

Type a request in plain language. Hera uses its tools, pausing for approval before anything
that edits files or runs a command (`[y]es / [a]lways / [n]o`). Read-only tools never prompt.

---

## Tools

| Tool | Purpose | Approval |
|---|---|---|
| `list_dir` | List a directory | auto |
| `read_file` | Read a file (line-numbered) | auto |
| `glob` | Find files by pattern (e.g. `**/*.py`) | auto |
| `search` | Grep file contents by regex | auto |
| `symbols` | Codebase index: list function/class/type definitions | auto |
| `semantic_search` | Embedding-ranked code search (only if an embeddings endpoint is reachable) | auto |
| `write_file` | Create / overwrite a file | prompts |
| `edit_file` | Exact string replacement in a file | prompts |
| `run_bash` | Run a shell command (offers to install a missing program, then retries) | prompts |
| `web_search` | Search the live web when Hera lacks info (auto-triggered) | auto |
| `web_fetch` | Fetch a page's readable text (e.g. a docs URL) | auto |
| `install_tool` | Download & install a program Hera decides it needs | prompts |
| `task` | Delegate a subtask to a focused sub-agent (own tool loop) | runs (inner calls prompt) |

Web tools let Hera look things up on its own when it's missing information; disable with
`HERA_NO_WEB=1`. Hera can also pull in tools it needs: it calls **`install_tool`** (you approve
first) to download a program it decides the task requires, or — if a `run_bash` command hits a
missing binary — it asks whether to install it (apt/dnf/pacman/brew/apk) and re-runs the
command. Either way nothing is downloaded without your OK. Disable with `HERA_NO_AUTOINSTALL=1`.
Pick `[a]lways` at a `write_file` prompt and Hera creates files on its own for the rest of the session.

## In-session commands

| Command | Effect |
|---|---|
| `/undo` | Revert the last file write/edit (repeatable) |
| `/diff` | Show the working-tree `git diff` |
| `/compact` | Summarize the conversation to free up context |
| `/tokens` | Show token usage this session |
| `/tools` | List the tools Hera can use (incl. MCP/custom) |
| `/allow` | List `run_bash` allow patterns (or `/allow <pattern>` to add one) |
| `/sandbox` | Show the `run_bash` sandbox status |
| `/sessions` | List saved sessions |
| `/resume` | Pick a past session from a list and resume it in place (`/resume <id>` to jump straight to one) |
| `/reasoning` | Toggle streaming of the model's thinking |
| `/cwd` | Show the working directory |
| `/new` | Save the current session and start a fresh one |
| `/clear` | Same as `/new` |
| `/help` | Show command list |
| `/exit` | Quit (Ctrl-C / Ctrl-D also work) |

Press **`/`** at the prompt to open a live command menu (Claude-Code style): it filters as you
type, `↑`/`↓` move the highlight, `Tab`/`Enter` accept, `Esc` dismisses; a bare `/` also lists
everything.

Reference files with **`@path`** to attach their contents. Every `write_file`/`edit_file` is
checkpointed, so **`/undo`** rolls back the last one.

## Sessions & resume

Conversations auto-save under `~/.config/hera/sessions/` after every turn:

```bash
hera --continue        # resume the most recent session
hera --resume <id>     # resume a specific session (id or prefix)
hera --list-sessions   # list saved sessions
```

`/sessions` lists them in-session and `/resume` lets you pick one to jump back into without
leaving Hera; `/new` saves the current one and starts fresh. Token totals
are restored on resume.

## Extending Hera — MCP & custom tools

Hera is an **MCP client**. List MCP servers in `~/.config/hera/mcp.json` (Claude-Desktop shape);
their tools appear as `mcp__<server>__<tool>`:

```json
{ "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"] }
} }
```

Or add **custom Python tools** at `~/.config/hera/tools.py` (global) or `.hera/tools.py`
(per-project):

```python
def _reverse(text=""): return text[::-1]
HERA_TOOLS = [{
    "name": "reverse_text", "description": "Reverse a string",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    "run": _reverse, "read_only": True,   # read_only → no approval prompt
}]
```

MCP and custom tools require approval unless `read_only`. Set `HERA_MCP_SANDBOX=1` to launch
MCP servers (third-party binaries) under the `run_bash` sandbox (fs confined to cwd, network
off with `bwrap`). Custom Python tools run in-process and aren't sandboxed — but you author
them, so they're trusted by construction; MCP servers are the untrusted surface.

**Sub-agents:** the `task` tool delegates a self-contained subtask to a focused sub-agent that
has the full toolset (minus `task`, so no recursion), shares the approval gate / sandbox /
checkpoints, and returns a concise result.

## Sandboxing & permissions

`run_bash` runs in a sandbox by default (`HERA_SANDBOX=auto`): **bubblewrap** if installed
(filesystem confined to the working dir, network off — best), else **`unshare`** on Linux
(PID-isolated, network off), else **none**. Set `HERA_SANDBOX_NET=1` to allow network (for
`pip install`, `git pull`, …), or `HERA_SANDBOX=none` to disable. Check with `/sandbox`.

Pre-approve safe commands so they don't prompt: `HERA_ALLOW="git status,git diff*,pytest*"`,
a `.heraallow` file (one pattern per line), `/allow <pattern>`, or **[a]/[p]** at a prompt. A
built-in denylist (`rm -rf /`, `sudo …`, `mkfs`, `curl … | sh`, …) always forces a prompt.

## Project context

If a `HERA.md` (or `AGENTS.md` / `AGENT.md`) file exists in the launch directory, Hera loads it
into its system prompt and follows its conventions — like Claude Code's `CLAUDE.md`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HERA_API_URL` | _(required)_ | Endpoint, e.g. `http://<host>:8080/v1` (given to you on approval). No host is baked into the code. |
| `HERA_API_KEY` | _(empty)_ | **Required** bearer key. Missing → `401`. |
| `HERA_USER` | _(key hash)_ | Identity for per-user session isolation (e.g. your email). |
| `HERA_MODEL` | `qwen3.6-35b-a3b` | Model name sent to the API |
| `HERA_NAME` | `Hera` | Assistant display name |
| `HERA_YOLO` | `0` | `1` = auto-approve every tool call (sandbox only) |
| `HERA_MAX_STEPS` | `25` | Max tool round-trips per message |
| `HERA_HIDE_REASONING` | `0` | `1` = don't stream the model's thinking |
| `HERA_NO_COLOR` | `0` | `1` = disable colour/styling (also honours `NO_COLOR`) |
| `HERA_FORCE_COLOR` | `0` | `1` = force colour even when output isn't a TTY |
| `HERA_SANDBOX` | `auto` | `run_bash` sandbox: `auto` / `bwrap` / `unshare` / `none` |
| `HERA_SANDBOX_NET` | `0` | `1` = allow network inside the sandbox |
| `HERA_ALLOW` | _(empty)_ | Comma-separated `run_bash` allow patterns (also reads `.heraallow`) |
| `HERA_DENY` | _(empty)_ | Extra deny patterns (added to the built-in list) |
| `HERA_SESSIONS_DIR` | `~/.config/hera/sessions` | Where session transcripts are saved |
| `HERA_MCP_CONFIG` | `~/.config/hera/mcp.json` | MCP servers config (Claude-Desktop shape) |
| `HERA_MCP_SANDBOX` | `0` | `1` = run MCP servers under the `run_bash` sandbox |
| `HERA_EMBED_URL` | = `HERA_API_URL` | Embeddings endpoint for `semantic_search` (server needs `--embeddings`) |
| `HERA_EMBED_MODEL` | = `HERA_MODEL` | Model name for embeddings requests |

> Legacy `QWEN_*` variables (and `LLAMA_API_KEY`) are honoured as fallbacks.
> `semantic_search` is enabled only when the embeddings endpoint responds; otherwise use `symbols` + `search`.

---

## Safety

`run_bash` and the file tools execute with **your** permissions in your current directory. The
approval prompts are the guardrail — read each one before approving. **Do not** set
`HERA_YOLO=1` outside a throwaway/sandbox directory.
