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

**Requirements:** Python 3.7+ (the installer adds the only dependency, `requests`).

### One command — the server's installer

The one-liner does everything: fetches `hera`, adds `requests`, puts it on your `PATH`, **and
saves the endpoint** so the only thing left is pasting your key:

```bash
HERA_SERVER=http://<HOST>:8081 bash <(curl -fsSL http://<HOST>:8081/install.sh)
hera        # first run: paste your Open WebUI API key once — done
```

That's the whole install — you don't also need the manual steps below.

<details>
<summary><b>Manual fallback</b> — straight from this public GitHub repo (no download server)</summary>

`hera.py` is a single file with one dependency:

```bash
pip install requests
curl -fsSL https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py -o ~/.local/bin/hera
chmod +x ~/.local/bin/hera
```

(Or `git clone https://github.com/jones0011738/hera-cli && chmod +x hera-cli/hera.py`.)
Make sure `~/.local/bin` is on your `PATH`. Optionally install `bubblewrap`
(`sudo apt install bubblewrap`) for full `run_bash` sandboxing.

Installed this way, no endpoint is saved yet, so first run also asks for it (the identity proxy,
`http://<HOST>:8090/v1`). To skip that prompt, write the file yourself:

```bash
mkdir -p ~/.config/hera
printf '{ "api_url": "http://<HOST>:8090/v1" }\n' > ~/.config/hera/config.json
hera        # paste your key once
```

Environment variables still work and **override** the file (handy for CI):

```bash
export HERA_API_URL=http://<HOST>:8090/v1   # the identity proxy
export HERA_API_KEY=sk-...                  # your Open WebUI API key
# HERA_USER is optional — the key's account email is resolved automatically.
```
</details>

**Your key is your identity:** on first run Hera asks the proxy which account the key belongs to,
greets you by **name + email** in the banner, and labels your sessions by that account — nothing
else to set (check with `hera whoami`). On a shared machine, **`hera logout`** (or `/logout`
in-session) clears the key + identity so a different user can paste their own key. This repo ships
**no key and no host** (that's why it can be public) — you supply both, once. See
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
| `run_bash` | Run a shell command (offers to install a missing program, then retries; `run_in_background` for servers/watchers) | prompts |
| `bash_output` / `bash_kill` | Read or stop a background `run_bash` job | auto / prompts |
| `todo_write` | Maintain the on-screen task checklist | auto |
| `web_search` | Search the live web when Hera lacks info (auto-triggered) | auto |
| `web_fetch` | Fetch a page's readable text (e.g. a docs URL) | auto |
| `install_tool` | Download & install a program Hera decides it needs | prompts |
| `task` | Delegate a subtask to a focused sub-agent (optional named `agent`) | runs (inner calls prompt) |

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
| `/compact` | Summarize the conversation to free up context (also automatic near the limit) |
| `/tokens` | Show token usage (and `$` cost, if priced) this session |
| `/plan` | Toggle plan mode — investigate & propose a plan before editing |
| `/auto` | Auto-approve level for this project: `read` / `edit` / `all` / `off` |
| `/todos` | Show the current task checklist |
| `/skills` | List the live shared-skills catalog (`/skills <id>` for detail) |
| `/tools` | List the tools Hera can use (incl. MCP/custom) |
| `/allow` | List `run_bash` allow patterns (or `/allow <pattern>` to add one) |
| `/sandbox` | Show the `run_bash` sandbox status |
| `/sessions` | List saved conversations (by their first message) |
| `/resume` | Pick a past conversation to resume — listed by its **first message**, choose by number |
| `/reasoning` | Toggle streaming of the model's thinking |
| `/cwd` | Show the working directory |
| `/new` | Save the current session and start a fresh one |
| `/clear` | Same as `/new` |
| `/logout` | Sign out and switch to a different API key (different user) |
| `/help` | Show command list |
| `/exit` | Quit (Ctrl-C / Ctrl-D also work) |

Press **`/`** at the prompt to open a live command menu (Claude-Code style): it filters as you
type, `↑`/`↓` move the highlight, `Tab`/`Enter` accept, `Esc` dismisses; a bare `/` also lists
everything.

Reference files with **`@path`** to attach their contents — or an **image** (`@shot.png`,
`.png/.jpg/.jpeg/.gif/.webp/.bmp`). The base model is text-only, so images are attached but only
*interpreted* when you set `HERA_VISION_URL` to a vision endpoint (image turns route there).
Every `write_file`/`edit_file` is checkpointed, so **`/undo`** rolls back the last one.

As Hera works it **narrates each step** (`→ Editing app.py`) above the tool card, alongside its
streaming reasoning. Before any edit it prints the **full proposed diff**, and the approval
prompt offers **`[t]ype feedback`** — pick it to send the model an instruction instead of a plain
yes/no. Press **`ESC`** while Hera is generating to **interrupt** the turn (history is kept).

## Sessions & resume

Conversations auto-save under `~/.config/hera/sessions/` after every turn. Inside a session,
**`/resume`** (or `/sessions`) lists the conversations **from the current project** by their
**first message** — pick one **by number**, no ID to remember (just like Claude Code):

```
Resume a conversation (this project, newest first)
   1. How do I add OAuth login to my Flask app
      2026-06-15 06:00 · 4 message(s)
   2. Refactor the auth middleware
      2026-06-15 05:00 · 2 message(s)
```

`/resume` and `/sessions` are **scoped to the project you launched in**. From the shell:
`hera --continue` reopens the latest **in this project**, `hera --list-sessions` lists **all**
projects, and `hera --resume <id>` still works for scripting. `/new` saves the current one and
starts fresh; token totals are restored on resume.

## Extending Hera — MCP & custom tools

Hera is an **MCP client** for both **local (stdio)** and **remote (HTTP/SSE)** servers. List them
in `~/.config/hera/mcp.json` (Claude-Desktop shape); their tools appear as `mcp__<server>__<tool>`:

```json
{ "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"] },
    "remote":     { "url": "https://mcp.example.com/mcp", "token": "${MY_MCP_TOKEN}" }
} }
```

A server with a **`command`** is launched locally over stdio; one with a **`url`** is reached over
the Streamable-HTTP transport (handles both JSON and SSE replies). For remote auth, `token`
(or `auth_token`) becomes `Authorization: Bearer …` — the credential an OAuth flow ultimately
issues (a personal-access/API token) — and a `headers` object sets anything else; both expand
`${ENV}` so secrets stay out of the file. (Interactive browser-based OAuth sign-in isn't supplied —
provide a token.)

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
checkpoints, and returns a concise result. Define **named agents** with their own instructions in
`~/.config/hera/agents/<name>.md` (optional `tools:` frontmatter to restrict the toolset) and
target one with `task(agent="<name>", …)`.

**Custom slash commands:** drop a `~/.config/hera/commands/<name>.md` file and run it as
`/<name> [args]`; `$ARGUMENTS` in the file is replaced with what you type.

## Auto-approve modes (per project)

`/auto` sets a Claude-Code-style auto-approve level, **remembered per project** and reversible any
time: `/auto read` (default — read-only tools auto-run, writes/commands prompt), `/auto edit`
(also auto-approve file writes/edits; shell still prompts), `/auto all` (auto-approve everything),
and `/auto off` to stop (back to `read`). `/auto` alone shows the current level. A `deny`
permission rule, **plan mode**, and `PreToolUse` hooks still take precedence even at `all`. Preset
with `HERA_AUTO_MODE=read|edit|all`.

## Plan mode, to-dos & cost

For multi-step work Hera keeps a live **to-do checklist** (`todo_write`; ○ → ▸ → ✔, reprint with
`/todos`) — the same pattern Claude Code uses — and prints a few **next-step suggestions** when a
task finishes.

**Plan mode** (`/plan` or `HERA_PLAN=1`) is the full Claude-Code flow: Hera researches read-only
(no edits/commands), then presents a plan and asks **“Ready to code?”** with `[1] yes, proceed`,
`[2] yes + auto-accept edits`, `[3] no, keep planning`. It implements only after you approve (and
leaves plan mode automatically); option 3 lets you give feedback and it re-plans.

When the conversation nears the context window it **auto-compacts** (tune with
`HERA_CONTEXT_TOKENS` / `HERA_AUTO_COMPACT_AT`). Set `HERA_PRICE_IN` / `HERA_PRICE_OUT` (USD per 1M
tokens) to see an estimated **`$` cost** in the summaries and `/tokens`.

## Sandboxing & permissions

`run_bash` runs in a sandbox by default (`HERA_SANDBOX=auto`): **bubblewrap** if installed
(filesystem confined to the working dir, network on by default — best), else **`unshare`** on Linux
(PID-isolated, network on), else **none**. Set `HERA_SANDBOX_NET=0` to block network again, or
`HERA_SANDBOX=none` to disable sandboxing entirely. Check with `/sandbox`.

Pre-approve safe commands so they don't prompt: `HERA_ALLOW="git status,git diff*,pytest*"`,
a `.heraallow` file (one pattern per line), `/allow <pattern>`, or **[a]/[p]** at a prompt. A
built-in denylist (`rm -rf /`, `sudo …`, `mkfs`, `curl … | sh`, …) always forces a prompt.

For finer control, `~/.config/hera/config.json` accepts a **`permissions`** block with per-tool
`allow`/`ask`/`deny` rules — e.g. `"deny": ["run_bash(rm *)"]`, `"allow": ["run_bash(git *)"]`,
`"ask": ["write_file"]` (`deny`/`ask` apply even under `HERA_YOLO`) — and a **`hooks`** block that
runs your own shell commands on `PreToolUse` (a non-zero exit blocks the tool), `PostToolUse`, and
`Stop`, each with an optional tool-name `matcher`.

## Project context

If a `HERA.md` (or `AGENTS.md` / `AGENT.md`) file exists in the launch directory, Hera loads it
into its system prompt and follows its conventions — like Claude Code's `CLAUDE.md`.

This stack also has a **server-side shared skills** layer: the identity proxy injects skills from
`shared-skills/skills/*.md` into CLI/VS Code/web requests using one shared catalog. Skills can
trigger automatically, or be forced with `@skill:<id>` / `/skill <id>`. Use `/skills` to inspect
the live catalog the proxy is serving.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HERA_API_URL` | _(required)_ | Endpoint, e.g. `http://<host>:8080/v1` (given to you on approval). No host is baked into the code. |
| `HERA_API_KEY` | _(empty)_ | **Required** bearer key. Missing → `401`. |
| `HERA_USER` | _(resolved from key)_ | Override the session label. Normally unset — the key's account email is fetched from the proxy automatically. |
| `HERA_MODEL` | `qwen3.6-35b-a3b` | Model name sent to the API |
| `HERA_NAME` | `Hera` | Assistant display name |
| `HERA_YOLO` | `0` | `1` = auto-approve every tool call (sandbox only) |
| `HERA_MAX_STEPS` | `25` | Max tool round-trips per message |
| `HERA_HIDE_REASONING` | `0` | `1` = don't stream the model's thinking |
| `HERA_PLAN` | `0` | `1` = start in plan mode (propose before editing) |
| `HERA_AUTO_MODE` | `read` | Auto-approve level: `read` / `edit` / `all` (per-project; `/auto` overrides & persists) |
| `HERA_NO_SUGGESTIONS` | `0` | `1` = don't print "Next steps" tips after a task |
| `HERA_PRICE_IN` / `HERA_PRICE_OUT` | `0` | USD per 1M input/output tokens → show `$` cost |
| `HERA_CONTEXT_TOKENS` / `HERA_AUTO_COMPACT_AT` | `32000` / `0.8` | Auto-compact history near the context window |
| `HERA_VISION_URL` | _(empty)_ | Vision endpoint for attached images. Unset → images attached but not interpreted (text-only model) |
| `HERA_VISION_MODEL` | = `HERA_MODEL` | Model name at `HERA_VISION_URL` |
| `HERA_NO_COLOR` | `0` | `1` = disable colour/styling (also honours `NO_COLOR`) |
| `HERA_FORCE_COLOR` | `0` | `1` = force colour even when output isn't a TTY |
| `HERA_SANDBOX` | `auto` | `run_bash` sandbox: `auto` / `bwrap` / `unshare` / `none` |
| `HERA_SANDBOX_NET` | `1` | `0` = block network inside the sandbox |
| `HERA_ALLOW` | _(empty)_ | Comma-separated `run_bash` allow patterns (also reads `.heraallow`) |
| `HERA_DENY` | _(empty)_ | Extra deny patterns (added to the built-in list) |
| `HERA_SESSIONS_DIR` | `~/.config/hera/sessions` | Where session transcripts are saved |
| `HERA_MCP_CONFIG` | `~/.config/hera/mcp.json` | MCP servers — local (`command`) or remote (`url` + bearer `token`) |
| `HERA_MCP_SANDBOX` | `0` | `1` = run local MCP servers under the `run_bash` sandbox |
| `HERA_EMBED_URL` | = `HERA_API_URL` | Embeddings endpoint for `semantic_search` (server needs `--embeddings`) |
| `HERA_EMBED_MODEL` | = `HERA_MODEL` | Model name for embeddings requests |
| `HERA_NO_UPDATE_CHECK` | `0` | `1` = don't check for or show the update notice |

> Legacy `QWEN_*` variables (and `LLAMA_API_KEY`) are honoured as fallbacks.
> `semantic_search` is enabled only when the embeddings endpoint responds; otherwise use `symbols` + `search`.

---

## Updating

Current release: **0.8.7**. On launch Hera checks the published version (at most once a day,
fail-silent) and prints a one-line notice when a newer one is out:

```
↑ update available: Hera 0.8.7 (you have 0.6.1)
  re-run the installer, or:  curl -fsSL <download_url> -o "$(command -v hera || echo ~/.local/bin/hera)"
```

**The simplest way is `hera doctor`** — it updates Hera in place to the latest version and then
runs a health check (endpoint, key, model, identity, sandbox). `hera doctor --force` re-downloads
even when already current. To update manually, re-run the one-line installer or pull the single
file directly:

```bash
curl -fsSL https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py -o "$(command -v hera || echo ~/.local/bin/hera)"
```

Your config and saved sessions are untouched. Silence the notice with `HERA_NO_UPDATE_CHECK=1`.

**Context limits are handled automatically.** When a long session or a large file read approaches
the model's context window, Hera auto-compacts the conversation (and self-heals a context-overflow
error by compacting and retrying), so you won't hit the old `400 … exceeds the available context
size` failure.

---

## Safety

`run_bash` and the file tools execute with **your** permissions in your current directory. The
approval prompts are the guardrail — read each one before approving. **Do not** set
`HERA_YOLO=1` outside a throwaway/sandbox directory.
