# Accessing the Hera CLI

**Hera** is the agentic terminal CLI for the Qwen3.6-35B-A3B model. It runs the model in a
reason→act loop with real tools — list/find/grep, read/write/edit files, run shell commands,
index symbols, semantic search, and delegate to sub-agents — **in the directory you launch it
from**. It asks for approval before edits/commands, sandboxes shell commands, streams the
model's reasoning, tracks tokens, and saves resumable per-user sessions.

> Other guides: [`ACCESS_VSCODE.md`](ACCESS_VSCODE.md) (VS Code / Cursor) ·
> [`ACCESS_WEB.md`](ACCESS_WEB.md) (Open WebUI) · [`README.md`](README.md) (full stack).
> Replace `<HOST>` below with the server address your admin gives you.

---

## 1. Get access (you need an account)

Hera is per-user: every user is a real **Open WebUI account** tied to your email, and the admin
must approve you.

1. Ask the admin to **create/approve your account** (signup is disabled).
2. Log in to the web UI at `http://<HOST>:3000`, then go to **Settings → Account → API keys**
   and **create an API key** (`sk-…`). This key *is* your CLI credential.

That key authenticates you on **both** the web UI and the CLI, and keeps your context separate
from everyone else's.

---

## 2. Install

**Requirements:** Python 3.7+ and the `requests` library — `hera.py` is one file, no other deps.

### Easiest — straight from the public GitHub repo

```bash
pip install requests
curl -fsSL https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py -o ~/.local/bin/hera
chmod +x ~/.local/bin/hera
```

Or clone it: `git clone https://github.com/jones0011738/hera-cli`. Ensure `~/.local/bin` is on
your `PATH`. Optionally install `bubblewrap` (`sudo apt install bubblewrap`) for full sandboxing.

> The repo ships **no key and no host** — that's why it can be public. You supply both in step 3.

### Alternative — the server's installer one-liner

If your admin runs the download server, this fetches `hera.py`, adds `requests`, and puts `hera`
on your `PATH`:

```bash
HERA_SERVER=http://<HOST>:8081 bash <(curl -fsSL http://<HOST>:8081/install.sh)
```

(Windows: install Python, `pip install requests`, download `hera.py` from the GitHub raw URL
above, run `python hera.py` — set the step-3 env vars in PowerShell with `$env:`.)

---

## 3. Configure & run

Point Hera at the **identity proxy** (port 8090) with your personal key:

```bash
export HERA_API_URL=http://<HOST>:8090/v1      # the identity proxy (validates your key)
export HERA_API_KEY=sk-...                     # your Open WebUI API key
export HERA_USER=you@example.com               # keeps your sessions separate
cd ~/my-project                                # Hera works in the current directory
hera
```

Add those `export` lines to your `~/.bashrc` to persist them. The proxy checks your key against
Open WebUI (must be an approved account), then forwards to the model — so you never handle a
shared server key, and usage is attributed to you.

> **No host or key is baked into the code** — that's why this CLI can live in a public repo and
> still work: you supply `HERA_API_URL` + `HERA_API_KEY`.

---

## 4. Using it

Type a request in plain language. Reference files with **`@path`** to attach them. Hera pauses
for approval before anything that edits files or runs a command:

```text
⚠ approval needed (run_bash)
  $ git diff
  [y]es once / [a]lways this command / [p]rogram (all 'git') / [n]o:
```

Read-only tools never prompt. `HERA_YOLO=1` auto-approves everything (sandbox/throwaway only).

### Tools

| Tool | Purpose | Approval |
|---|---|---|
| `list_dir`, `read_file` | List a dir / read a file | auto |
| `glob` | Find files by pattern (`**/*.py`) | auto |
| `search` | Grep file contents by regex | auto |
| `symbols` | Codebase index of definitions | auto |
| `semantic_search` | Embedding-ranked code search (if enabled) | auto |
| `write_file`, `edit_file` | Create / edit a file | prompts |
| `run_bash` | Run a shell command (sandboxed) | prompts |
| `task` | Delegate a subtask to a sub-agent | runs (inner calls prompt) |

### In-session commands

| Command | Effect |
|---|---|
| `/undo` | Revert the last file write/edit |
| `/diff` | Show the working-tree `git diff` |
| `/compact` | Summarize the conversation to free context |
| `/tokens` | Token usage this session |
| `/tools` | List tools (incl. MCP/custom) |
| `/allow [pat]` | List or add `run_bash` allow patterns |
| `/sandbox` | Show the sandbox status |
| `/sessions` | List saved sessions |
| `/reasoning` | Toggle streaming the model's thinking |
| `/cwd` | Show the working directory |
| `/new`, `/clear` | Start a fresh session |
| `/help`, `/exit` | Help / quit |

### Sessions & resume

Conversations auto-save under `~/.config/hera/sessions/<you>/`. Resume with
`hera --continue`, `hera --resume <id>`, or `hera --list-sessions`. Because the store is
namespaced by `HERA_USER` (or a hash of your key), users never share context on one machine.

### Sandboxing & permissions

`run_bash` runs sandboxed by default: **bubblewrap** (filesystem confined to the working dir,
network off) if installed, else `unshare` (PID/network isolation), else none. `HERA_SANDBOX_NET=1`
allows network (for `pip install`, `git pull`); `/sandbox` shows the active level. Pre-approve
safe commands with `HERA_ALLOW`, a `.heraallow` file, `/allow`, or `[a]`/`[p]` at a prompt; a
built-in denylist always forces a prompt for dangerous commands.

### Project context

A `HERA.md` (or `AGENTS.md`/`AGENT.md`) in the launch directory is loaded into the system prompt
— like Claude Code's `CLAUDE.md`.

---

## 5. Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `HERA_API_URL` | _(required)_ | Endpoint — the identity proxy `http://<HOST>:8090/v1`. |
| `HERA_API_KEY` | _(required)_ | Your Open WebUI API key. Missing/invalid → `401`. |
| `HERA_USER` | _(key hash)_ | Identity for per-user session isolation (e.g. your email). |
| `HERA_MODEL` | `qwen3.6-35b-a3b` | Model name. |
| `HERA_YOLO` | `0` | `1` = auto-approve every tool call. Sandbox only. |
| `HERA_MAX_STEPS` | `25` | Max tool round-trips per message. |
| `HERA_HIDE_REASONING` | `0` | `1` = don't stream the model's thinking. |
| `HERA_NO_COLOR` / `HERA_FORCE_COLOR` | `0` | Disable / force colour. |
| `HERA_SANDBOX` | `auto` | `auto` / `bwrap` / `unshare` / `none`. |
| `HERA_SANDBOX_NET` | `0` | `1` = allow network in the sandbox. |
| `HERA_ALLOW` / `HERA_DENY` | _(empty)_ | `run_bash` allow / extra-deny patterns. |
| `HERA_MCP_CONFIG` | `~/.config/hera/mcp.json` | MCP servers (Claude-Desktop shape). |
| `HERA_MCP_SANDBOX` | `0` | `1` = run MCP servers under the sandbox. |
| `HERA_EMBED_URL` / `HERA_EMBED_MODEL` | = API | Embeddings endpoint for `semantic_search`. |
| `HERA_SESSIONS_DIR` | `~/.config/hera/sessions` | Session store root (namespaced per user). |

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `401 Unauthorized` | Key wrong/expired, or account not approved. Re-create your key in the web UI; ask the admin to approve you. |
| `no server set` | Export `HERA_API_URL` (the `:8090/v1` proxy). |
| `Cannot reach …:8090` | Proxy down or firewall. Check `curl http://<HOST>:8090/health`. |
| `'requests' not found` | `pip install requests`. |
| Long silent `thinking…` | It's a reasoning model; `/reasoning` toggles visibility or set `HERA_HIDE_REASONING=1`. |

---

## Safety note

`run_bash` and the file tools act with **your** permissions in your working directory. The
approval prompts (and the sandbox) are the guardrails — read each prompt before approving, and
don't set `HERA_YOLO=1` outside a throwaway directory.
