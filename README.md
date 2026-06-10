# Hera CLI

**Hera** is a single-file, agentic coding CLI for the Qwen3.6-35B-A3B model (served via an
OpenAI-compatible llama.cpp endpoint). It runs the model in a reason→act loop with real tools:
it lists directories, finds and greps code, reads/writes/edits files, and runs shell commands
**in the directory you launch it from**. It asks for approval before editing files or running
commands, streams the model's reasoning, and tracks token usage.

> Single file, no framework. Requires Python 3.7+ and `requests`.
> This repo is kept in sync from the gpustack stack; see `scripts/sync-hera-cli.sh` there.

---

## Install

> Replace `<SERVER_IP>` with the server's host address (provided separately by the admin). The
> downloaded `hera` and `install.sh` already point at the right server by default.

**One line** (downloads `hera` to `~/.local/bin`):

```bash
curl -fsSL http://<SERVER_IP>:8081/install.sh | bash
```

The installer is plain, inspectable text — read it first if you like:
`curl -fsSL http://<SERVER_IP>:8081/install.sh`. It needs no root and sends nothing anywhere.

**Manual:**

```bash
pip install requests
curl http://<SERVER_IP>:8081/hera.py -o ~/.local/bin/hera   # or copy hera.py from this repo
chmod +x ~/.local/bin/hera
```

Make sure `~/.local/bin` is on your `PATH`.

---

## Run

```bash
export HERA_API_KEY=<key>     # bearer key the server enforces
cd ~/my-project               # Hera works in the current directory
hera
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
| `run_bash` | Run a shell command | prompts |
| `task` | Delegate a subtask to a focused sub-agent (own tool loop) | runs (inner calls prompt) |

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
| `/reasoning` | Toggle streaming of the model's thinking |
| `/cwd` | Show the working directory |
| `/new` | Save the current session and start a fresh one |
| `/clear` | Same as `/new` |
| `/help` | Show command list |
| `/exit` | Quit (Ctrl-C / Ctrl-D also work) |

Reference files with **`@path`** to attach their contents. Every `write_file`/`edit_file` is
checkpointed, so **`/undo`** rolls back the last one.

## Sessions & resume

Conversations auto-save under `~/.config/hera/sessions/` after every turn:

```bash
hera --continue        # resume the most recent session
hera --resume <id>     # resume a specific session (id or prefix)
hera --list-sessions   # list saved sessions
```

`/sessions` lists them in-session; `/new` saves the current one and starts fresh. Token totals
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
| `HERA_API_KEY` | _(empty)_ | **Required** bearer key. Missing → `401`. |
| `HERA_API_URL` | `http://<SERVER_IP>:8080/v1` | Server URL |
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
