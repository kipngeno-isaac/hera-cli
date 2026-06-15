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

## 0. What you need first

| Thing | Value |
|---|---|
| **`<HOST>`** | The server address your admin gives you (IP or hostname). Substitute it everywhere `<HOST>` appears. |
| **An approved account + API key** | A `user`/`admin` Open WebUI account and a personal `sk-…` key (see step 1). |
| **Python 3.7+** | The only runtime requirement. The installer adds the one library (`requests`). |
| **`bubblewrap`** *(optional)* | Linux-only; gives `run_bash` full filesystem confinement. Without it Hera falls back to a weaker sandbox. |

Ports you may touch: **`:8081`** download host (installer + `hera.py`), **`:8090`** the identity
proxy Hera actually calls (`/v1`, `/whoami`, `/health`), **`:3000`** the web UI where you mint your
key. The installer derives `:8090` from the `:8081` host automatically, so normally you only ever
type the `:8081` one-liner.

### TL;DR (already approved, Linux/macOS)

```bash
HERA_SERVER=http://<HOST>:8081 bash <(curl -fsSL http://<HOST>:8081/install.sh)
cd ~/my-project && hera        # paste your sk-… key once — done
```

The rest of this guide explains each step, Windows/offline installs, and every option.

---

## 1. Get access (you need an account)

Hera is per-user: every user is a real **Open WebUI account** tied to your email, and the admin
must approve you.

1. Ask the admin to **create/approve your account** (signup is disabled).
2. Log in to the web UI at `http://<HOST>:3000`, then go to **Settings → Account → API keys**
   and **create an API key** (`sk-…`). This key *is* your CLI credential.

That key authenticates you on **both** the web UI and the CLI, and keeps your context separate
from everyone else's.

> **Sanity-check the server is reachable** before installing (optional):
> ```bash
> curl http://<HOST>:8090/health           # → {"status": "ok"}
> curl -H "Authorization: Bearer sk-…" http://<HOST>:8090/whoami   # → {"email": "...", "role": "user", "name": "Your Name"}
> ```
> A `200` from `/health` means the proxy is up; a good `/whoami` means your key is valid **and**
> approved. A `403` there means your account is still `pending` (ask the admin); `401` means a bad key.

---

## 2. Install — one command

**Requirements:** Python 3.7+ (the installer adds the only dependency, `requests`).

One line does everything — checks Python, installs `requests`, downloads `hera` onto your
`PATH`, **and pre-configures the endpoint** so the only thing left is pasting your key:

```bash
HERA_SERVER=http://<HOST>:8081 bash <(curl -fsSL http://<HOST>:8081/install.sh)
```

You'll see (verified output):

```
Hera CLI installer
  download from : http://<HOST>:8081/hera.py
  install to    : ~/.local/bin/hera

✓ python3: Python 3.10.12
✓ requests already installed
✓ bubblewrap present — run_bash gets full filesystem confinement
• downloading hera…
✓ installed: ~/.local/bin/hera
✓ endpoint saved: http://<HOST>:8090/v1  (~/.config/hera/config.json)

Done. The endpoint is already configured. Just run:

       hera
```

That's the whole install. Skip to step 3 — you do **not** also need the `pip`/`curl`/`chmod`
steps below; the installer already did them. (If `~/.local/bin` isn't on your `PATH`, the
installer prints the one line to add it.)

<details>
<summary><b>Manual fallback</b> (offline, Windows, or no download server)</summary>

`hera.py` is a single file with one dependency. Grab it straight from the public GitHub repo:

```bash
pip install requests
curl -fsSL https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py -o ~/.local/bin/hera
chmod +x ~/.local/bin/hera
```

Or clone it: `git clone https://github.com/jones0011738/hera-cli`. Ensure `~/.local/bin` is on
your `PATH`. Optionally install `bubblewrap` (`sudo apt install bubblewrap`) for full sandboxing.
The repo ships **no key and no host** (that's why it can be public) — so installed this way Hera
also asks for the endpoint on first run (the identity proxy, `http://<HOST>:8090/v1`).

**Windows** (no `bash`/`curl <(…)`, so the one-liner doesn't apply — do it manually):

```powershell
py -m pip install requests
# download hera.py somewhere on your PATH, e.g. into a folder you control:
curl.exe -fsSL https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py -o hera.py
py hera.py
```

On Windows the config lives at `%USERPROFILE%\.config\hera\config.json` and sessions under
`%USERPROFILE%\.config\hera\sessions\`. `bubblewrap` doesn't exist on Windows, so `run_bash`
runs with `HERA_SANDBOX=none` (unconfined) — keep the approval prompts on. The first run still
asks for the endpoint (`http://<HOST>:8090/v1`) and your key, then saves both.

**macOS:** same one-liner as Linux. There's no `bubblewrap` on macOS, so `run_bash` uses the
weaker fallback or `none`; everything else is identical.
</details>

---

## 3. Run — just paste your key once

If you used the **installer one-liner** above, the endpoint is already saved. Just run:

```bash
cd ~/my-project        # Hera works in the current directory
hera
```

On first launch Hera asks you to **paste your API key** once, saves it to
`~/.config/hera/config.json` (mode 600), and never asks again (verified output):

```
▌ Welcome to Hera  · one-time setup

  Your personal API key from Open WebUI → Settings → Account → API Keys.
  Paste your API key: sk-…………………

✓ saved to ~/.config/hera/config.json — you're set; this won't ask again.
✓ signed in as Your Name (you@example.com) — sessions are labelled by your account.
```

…then the banner appears and you're at the prompt.

**Your key is your identity.** Right after you paste it, Hera asks the proxy's `/whoami` which
account the key belongs to and shows your **name + email** in the banner, labelling your sessions by
that account automatically — there's nothing else to set. The same config also powers the VS Code
extension. (Check anytime with `hera whoami`.)

**Switching users.** On a shared machine, sign out and hand over with **`hera logout`** (or
**`/logout`** inside a session). It clears the saved key and identity — keeping the endpoint — so
the next person just pastes *their* key and gets their own name, sessions, and history. In VS Code,
clear (or change) the `hera.apiKey` setting and reload the panel.

> **Installed manually (raw GitHub), so no endpoint is saved yet?** Hera will ask for the
> endpoint too on first run (it's the identity proxy, `http://<HOST>:8090/v1`). Or set it once:
> ```bash
> mkdir -p ~/.config/hera
> printf '{ "api_url": "http://<HOST>:8090/v1" }\n' > ~/.config/hera/config.json
> ```

> **Prefer environment variables?** They still work and **override** the config file — handy for
> CI or throwaway shells:
> ```bash
> export HERA_API_URL=http://<HOST>:8090/v1   # identity proxy
> export HERA_API_KEY=sk-...                  # your Open WebUI API key
> # HERA_USER is optional — only to force a label; normally the key resolves it.
> ```

The proxy checks your key against Open WebUI (must be an **approved**, non-`pending` account),
then forwards to the model — so you never handle a shared server key, and usage is attributed to
you.

> **No host or key is baked into the code** — that's why this CLI can live in a public repo and
> still work: you supply the endpoint + key once, on your machine.

---

## 4. Using it

Type a request in plain language. Reference files with **`@path`** to attach them — including
**images** (`@screenshot.png`; see "Images" below). As Hera works it narrates each step in plain
language (`→ Reading config.py`) above the tool card, alongside its streaming reasoning, so you
can follow what it's doing. Hera pauses for approval before anything that edits files or runs a
command, and for edits it shows the **full proposed diff first**:

```text
⚠ approval needed (edit_file)
  edit config.py
    @@ -10,3 +10,3 @@
     def load():
    -    key = os.environ["KEY"]
    +    key = os.environ.get("KEY", "")
  run this? [y]es / [a]lways / [t]ype feedback / [n]o:
```

Choose **`[t]`** to type an instruction instead of a plain yes/no — your text goes straight back
to the model ("no, use `getpass` instead"). Press **`ESC`** at any time while Hera is generating
to **interrupt** the current turn; your session and history are kept, and you're returned to the
prompt (Ctrl-C still works too).

Read-only tools never prompt. `HERA_YOLO=1` auto-approves everything (sandbox/throwaway only).

### Images

Attach an image with `@path` (`.png/.jpg/.jpeg/.gif/.webp/.bmp`):

```text
❯ what's wrong here? @error-screenshot.png
```

The default Qwen model is **text-only**, so out of the box Hera attaches the image but tells you
it can't interpret it. To actually analyze images, point Hera at a vision-capable
OpenAI-compatible endpoint with `HERA_VISION_URL` (and `HERA_VISION_MODEL`); image turns are then
routed there automatically.

### Tools

| Tool | Purpose | Approval |
|---|---|---|
| `list_dir`, `read_file` | List a dir / read a file | auto |
| `glob` | Find files by pattern (`**/*.py`) | auto |
| `search` | Grep file contents by regex | auto |
| `symbols` | Codebase index of definitions | auto |
| `semantic_search` | Embedding-ranked code search (if enabled) | auto |
| `write_file`, `edit_file` | Create / edit a file | prompts |
| `run_bash` | Run a shell command (sandboxed; `run_in_background` for servers/watchers) | prompts |
| `bash_output`, `bash_kill` | Read / stop a background `run_bash` job | auto / prompts |
| `todo_write` | Maintain the on-screen task checklist | auto |
| `task` | Delegate a subtask to a sub-agent (optional named `agent`) | runs (inner calls prompt) |

### In-session commands

| Command | Effect |
|---|---|
| `/undo` | Revert the last file write/edit |
| `/diff` | Show the working-tree `git diff` |
| `/compact` | Summarize the conversation to free context (also automatic near the limit) |
| `/tokens` | Token usage (and `$` cost, if priced) this session |
| `/plan` | Toggle plan mode — investigate & propose before editing |
| `/auto` | Set auto-approve level for this project: `read` / `edit` / `all` / `off` |
| `/todos` | Show the current task checklist |
| `/skills` | List the live shared-skills catalog (`/skills <id>` for detail) |
| `/tools` | List tools (incl. MCP/custom) |
| `/allow [pat]` | List or add `run_bash` allow patterns |
| `/sandbox` | Show the sandbox status |
| `/sessions` | List saved conversations (by their first message) |
| `/resume` | Pick a past conversation to resume (by its first message) |
| `/reasoning` | Toggle streaming the model's thinking |
| `/cwd` | Show the working directory |
| `/new`, `/clear` | Start a fresh session |
| `/logout` | Sign out and switch to a different API key (a different user) |
| `/help`, `/exit` | Help / quit |

### Sessions & resume

Conversations auto-save under `~/.config/hera/sessions/<you>/`. The friendliest way back in is
**`/resume`** — like Claude Code, it lists the conversations **from the project you're in** by their
**first question**, and you pick one **by number**; no ID to remember:

```
Resume a conversation (this project, newest first)
   1. How do I add OAuth login to my Flask app
      2026-06-15 06:00 · 4 message(s)
   2. Refactor the auth middleware
      2026-06-15 05:00 · 2 message(s)

  number to resume (Enter to cancel):
```

`/resume` and `/sessions` are **scoped to the current project** (the directory you launched in), so
you only see *this* project's history (with a note if you have conversations elsewhere). `hera
--continue` reopens the latest **in this project**; `hera --list-sessions` shows **all** projects,
and `hera --resume <id>` still works for any conversation. The store is namespaced by the account
your key resolves to, so users never share context on one machine.

### Auto-approve modes (per project)

Tired of approving every step? Set an **auto mode** with `/auto`, just like Claude Code. It is
**remembered per project** (keyed by the directory you launch in) and you can **stop it any time**:

| `/auto` … | What runs without asking |
|---|---|
| `/auto read` *(default)* | Read-only tools only — every write/command still prompts |
| `/auto edit` | Reads **and** file writes/edits — shell commands still prompt |
| `/auto all` | **Everything** — full auto for this project |
| `/auto off` | Stop auto mode (back to `read`) |

`/auto` with no argument shows the current level. Safety always wins over auto mode: a `deny`
permission rule, **plan mode**, and `PreToolUse` hooks still block or prompt even at `all`. (You
can also preset it with `HERA_AUTO_MODE=read|edit|all`.)

### Plan mode, to-dos, and next-step tips

Like Claude Code, Hera lays out a **to-do checklist** (`todo_write`) for multi-step tasks and
updates it live (○ pending → ▸ in-progress → ✔ done; `/todos` reprints it). After a task that used
tools, it prints a few **next-step suggestions**.

**Plan mode** (`/plan` or `HERA_PLAN=1`) is the full Claude-Code flow: Hera investigates
**read-only** — no edits, commands, or installs — then presents a plan and asks you to proceed:

```
▌ Ready to code?  Here's the plan:
  1. Add the /auth/callback route in app.py
  2. Store the token in the session
  3. Add a logout button to the navbar

  [1] yes, proceed   [2] yes + auto-accept edits   [3] no, keep planning:
```

Pick **1** to approve and let it implement (it leaves plan mode automatically), **2** to approve
*and* switch this project to auto-accept edits (`/auto edit`), or **3** to keep refining — type
feedback and it re-plans. Nothing is changed until you approve. Toggle the mode anytime with
`/plan`.

### Sandboxing & permissions

`run_bash` runs sandboxed by default: **bubblewrap** (filesystem confined to the working dir,
network on) if installed, else `unshare` (PID isolation, network on), else none. `HERA_SANDBOX_NET=0`
turns network back off if you want a stricter shell sandbox; `/sandbox` shows the active level. Pre-approve
safe commands with `HERA_ALLOW`, a `.heraallow` file, `/allow`, or `[a]`/`[p]` at a prompt; a
built-in denylist always forces a prompt for dangerous commands.

For finer control, `~/.config/hera/config.json` accepts a **`permissions`** block with
`allow`/`ask`/`deny` rules per tool — e.g. `"deny": ["run_bash(rm *)"]`,
`"allow": ["run_bash(git *)"]`, `"ask": ["write_file"]` (`deny`/`ask` apply even under YOLO) — and a
**`hooks`** block running your own commands on `PreToolUse` (a non-zero exit blocks the tool),
`PostToolUse`, and `Stop`. With pricing set (`HERA_PRICE_IN`/`HERA_PRICE_OUT`, USD per 1M tokens),
summaries and `/tokens` show an estimated **`$` cost**.

### Project context

A `HERA.md` (or `AGENTS.md`/`AGENT.md`) in the launch directory is loaded into the system prompt
— like Claude Code's `CLAUDE.md`.

This stack also has a **server-side shared skills** layer: the identity proxy injects skills from
[`shared-skills/skills/`](shared-skills/skills/) into CLI/VS Code chat requests, using the same
registry the browser chat uses. Users can activate a skill explicitly with `@skill:<id>` or
`/skill <id>`.

Use **`/skills`** to see the live server-side catalog, or **`/skills <id>`** for one skill's
details. That output comes from the proxy, so it reflects the same enabled/disabled state the web
chat is using.

---

## 5. Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `HERA_API_URL` | _(required)_ | Endpoint — the identity proxy `http://<HOST>:8090/v1`. |
| `HERA_API_KEY` | _(required)_ | Your Open WebUI API key. Missing/invalid → `401`. |
| `HERA_USER` | _(resolved from key)_ | Override the session label. Normally unset — the key's account email is fetched from the proxy automatically. |
| `HERA_MODEL` | `qwen3.6-35b-a3b` | Model name. |
| `HERA_YOLO` | `0` | `1` = auto-approve every tool call. Sandbox only. |
| `HERA_MAX_STEPS` | `25` | Max tool round-trips per message. |
| `HERA_HIDE_REASONING` | `0` | `1` = don't stream the model's thinking. |
| `HERA_PLAN` | `0` | `1` = start in plan mode (propose before editing). |
| `HERA_AUTO_MODE` | `read` | Auto-approve level: `read` / `edit` / `all` (per-project default; `/auto` overrides & persists). |
| `HERA_NO_SUGGESTIONS` | `0` | `1` = don't print "Next steps" tips after a task. |
| `HERA_PRICE_IN` / `HERA_PRICE_OUT` | `0` | USD per 1M input/output tokens → show `$` cost. |
| `HERA_CONTEXT_TOKENS` / `HERA_AUTO_COMPACT_AT` | `32000` / `0.8` | Auto-compact history near the context window. |
| `HERA_VISION_URL` | _(empty)_ | Vision endpoint for attached images. Unset → images attached but not interpreted (text-only model). |
| `HERA_VISION_MODEL` | = `HERA_MODEL` | Model name at `HERA_VISION_URL`. |
| `HERA_NO_COLOR` / `HERA_FORCE_COLOR` | `0` | Disable / force colour. |
| `HERA_SANDBOX` | `auto` | `auto` / `bwrap` / `unshare` / `none`. |
| `HERA_SANDBOX_NET` | `1` | `0` = block network in the sandbox. |
| `HERA_ALLOW` / `HERA_DENY` | _(empty)_ | `run_bash` allow / extra-deny patterns. |
| `HERA_MCP_CONFIG` | `~/.config/hera/mcp.json` | MCP servers — local (`command`) or remote (`url` + bearer `token`). |
| `HERA_MCP_SANDBOX` | `0` | `1` = run local MCP servers under the sandbox. |
| `HERA_EMBED_URL` / `HERA_EMBED_MODEL` | = API | Embeddings endpoint for `semantic_search`. |
| `HERA_SESSIONS_DIR` | `~/.config/hera/sessions` | Session store root (namespaced per user). |
| `HERA_NO_UPDATE_CHECK` | `0` | `1` = don't check for or show the update notice. |

---

## 6. Keeping Hera up to date

The current release is **0.8.7**. On launch Hera checks the published version (at most once a
day, fail-silent — it never blocks or errors startup). If a newer one is out, you'll see a
one-line notice like:

```
↑ update available: Hera 0.8.7 (you have 0.6.1)
  re-run the installer, or:  curl -fsSL <download_url> -o "$(command -v hera || echo ~/.local/bin/hera)"
```

**Easiest — just run `hera doctor`.** It updates Hera to the latest version in place and then runs
a quick health check (endpoint, key, model, identity, sandbox):

```
$ hera doctor

▌ Hera doctor  · update + health check

  ✓ update       updated v0.6.1 → v0.8.7  ·  ~/.local/bin/hera
  ✓ endpoint     http://<HOST>:8090/v1
  ✓ api key      set
  ✓ model        qwen3.6-35b-a3b — HTTP 200
  ✓ identity     Your Name (you@example.com)
  ✓ sandbox      bwrap — fs confined to cwd, network on
  ✓ context      auto-compacts near 32000 tok, and self-recovers on overflow
```

The **identity** line shows the real account the key belongs to (your name + email, fetched live
from `/whoami`) — so every surface greets *you*. `hera doctor --force` re-downloads even if you're
already current. Or update manually — re-run the one-line installer from step 2, or pull the latest
single file:

```bash
curl -fsSL http://<HOST>:8081/hera.py -o "$(command -v hera || echo ~/.local/bin/hera)"
# raw-GitHub installs instead use: https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py
```

Your config and saved sessions are untouched by an update. Set `HERA_NO_UPDATE_CHECK=1` to silence
the notice.

> **No more "context size exceeded" errors.** If a long session or a big file read fills the
> 32k-token window, Hera now **auto-compacts the history and retries** instead of failing the turn —
> so the old `400 Bad Request … exceeds the available context size` is handled for you.

---

## 7. Troubleshooting

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
