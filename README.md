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
| `write_file` | Create / overwrite a file | prompts |
| `edit_file` | Exact string replacement in a file | prompts |
| `run_bash` | Run a shell command | prompts |

## In-session commands

| Command | Effect |
|---|---|
| `/tokens` | Show token usage this session |
| `/tools` | List the tools Hera can use |
| `/allow` | List `run_bash` allow patterns (or `/allow <pattern>` to add one) |
| `/sandbox` | Show the `run_bash` sandbox status |
| `/reasoning` | Toggle streaming of the model's thinking |
| `/cwd` | Show the working directory |
| `/clear` | Erase conversation and session approvals |
| `/help` | Show command list |
| `/exit` | Quit (Ctrl-C / Ctrl-D also work) |

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
| `HERA_SANDBOX` | `auto` | `run_bash` sandbox: `auto` / `bwrap` / `unshare` / `none` |
| `HERA_SANDBOX_NET` | `0` | `1` = allow network inside the sandbox |
| `HERA_ALLOW` | _(empty)_ | Comma-separated `run_bash` allow patterns (also reads `.heraallow`) |
| `HERA_DENY` | _(empty)_ | Extra deny patterns (added to the built-in list) |

> Legacy `QWEN_*` variables (and `LLAMA_API_KEY`) are honoured as fallbacks.

---

## Safety

`run_bash` and the file tools execute with **your** permissions in your current directory. The
approval prompts are the guardrail — read each one before approving. **Do not** set
`HERA_YOLO=1` outside a throwaway/sandbox directory.
