# Accessing Hera in VS Code / Cursor

The **Hera extension** brings the agent into your editor: a native **chat panel** (streaming
answers + thinking, step-by-step narration, tool cards, inline approval buttons with
**reject-with-feedback** and **Stop**, and **image attachments**), **in-editor diffs of the
proposed change before it's applied**, and **LSP-driven context** (send the file's symbols +
diagnostics with your question). It drives the same `hera` CLI under the hood — one agent, just a
graphical surface.

> Cursor is a VS Code fork, so everything here works in Cursor identically.
> See also [`ACCESS_CLI.md`](ACCESS_CLI.md) (terminal) and [`ACCESS_WEB.md`](ACCESS_WEB.md) (web).
> Replace `<HOST>` with the server address your admin gives you.

---

## 1. Prerequisites

1. **An approved account + API key.** Log in at `http://<HOST>:3000` and create an API key under
   **Settings → Account → API keys** (ask the admin to approve your account first).
2. **The `hera` CLI installed** and on your `PATH` (the extension shells out to it):
   ```bash
   HERA_SERVER=http://<HOST>:8081 bash <(curl -fsSL http://<HOST>:8081/install.sh)
   ```
   (Or set `hera.command` in the extension settings to an absolute path.)
3. **Credentials.** Easiest: **run `hera` once in a terminal and paste your key** (see
   [`ACCESS_CLI.md`](ACCESS_CLI.md)). That saves `~/.config/hera/config.json` (endpoint + key +
   your resolved account email), and `hera --serve` — what the extension drives — reads that file
   automatically, so the extension just works with no settings.
   - Prefer to keep it all in the editor? Set `hera.serverUrl` + `hera.apiKey` in the extension
     settings instead (below). These override the config file.
4. **`bubblewrap`** (optional, recommended) for full shell sandboxing.

---

## 2. Install the extension

The extension lives in `ide/vscode-hera/` (in the full stack repo). It's a scaffold-grade extension
distributed as source (no Marketplace listing), so you load it locally:

**Run from source (quickest):**
1. Open the `ide/vscode-hera/` folder in VS Code / Cursor.
2. Press **F5** → *Run Extension*. A second editor window opens with Hera loaded.

**Or package + install a `.vsix`:**
```bash
cd ide/vscode-hera
npm install -g @vscode/vsce
vsce package                       # produces hera-cli-0.2.0.vsix
```
Then in VS Code: **Extensions → ⋯ → Install from VSIX…** and pick the file.

---

## 3. Configure

Open **Settings → Extensions → Hera** (or edit `settings.json`):

```json
{
  "hera.command": "hera",
  "hera.serverUrl": "http://<HOST>:8090/v1",
  "hera.apiKey": "sk-your-open-webui-key",
  "hera.showDiffs": true,
  "hera.visionUrl": ""
}
```

- All of these are **optional if you've already run `hera` once** — the saved config supplies them.
- `hera.serverUrl` → the **identity proxy** (`:8090/v1`).
- `hera.apiKey` → your personal Open WebUI key (or leave blank to use the saved config / your
  shell's `HERA_API_KEY`).
- Your key is your identity — sessions are labelled by the account it resolves to, so there's no
  `HERA_USER` to set. Use `hera.extraEnv` only for extras like `HERA_YOLO` or `HERA_EMBED_URL`.

---

## 4. Using it

| Command (Command Palette / keys) | What it does |
|---|---|
| **Hera: Open chat panel** (`Ctrl/Cmd+Alt+H`) | Native chat webview — streaming, tool cards, approval buttons |
| **Hera: Ask with file context (LSP)** | Sends your question + the active file's symbols & diagnostics |
| **Hera: Ask about current file / selection** | Quick prompt with `@file` / a line range (right-click menu too) |
| **Hera: Start terminal session** | Runs `hera` in an integrated terminal (full TUI) |
| **Hera: Resume last session** | `hera --continue` in a terminal |

**The chat panel:**
- The model's **thinking** streams in a block that's open while it works, then collapses; before
  each tool call a plain-language **`→` narration** line says what Hera is about to do; the
  **answer** renders as markdown.
- Each tool call shows as a card (`◆ tool → result`). When Hera is about to **edit a file**, a
  native **diff editor opens showing the proposed change before it's applied** (before ↔ after),
  so you review it *before* approving. Toggle with `hera.showDiffs`.
- When a write/edit/shell command needs approval, **buttons** appear inline
  (*Approve once / Always / Deny*) plus a text box and **Reject with feedback** — type what Hera
  should do instead and it goes straight back to the model.
- While Hera is generating, the **Send** button becomes **Stop** — click it to interrupt the turn.
- **Attach an image** with the **📎** button, or **paste**/**drag-drop** one into the message box;
  it shows as a thumbnail chip. The base model is text-only, so set `hera.visionUrl` (a vision
  endpoint) to actually analyze images — otherwise they're attached but not interpreted.
- Per-turn token usage is shown.

---

## 5. How it works

```
 chat webview  ⇄  extension  ⇄  hera --serve  ⇄  identity proxy (:8090)  ⇄  model
                                  (full agent: tools, sandbox, sessions, MCP)
```

The extension spawns `hera --serve` (a headless JSON mode); the CLI streams JSON events the
webview renders, and your replies/approvals go back over stdin. Same agent as the terminal,
same per-user auth via your Open WebUI key.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| "could not start 'hera'" | Install the CLI / set `hera.command` to its absolute path. |
| Chat shows `⚠ no API key` | Set `hera.apiKey` (or `HERA_API_KEY` in your shell). |
| `401` / nothing comes back | Key wrong/expired or account not approved; recreate the key, ask admin to approve. |
| No diff opens on edits | Set `hera.showDiffs: true`. |
| "Ask with context" empty | The file's language needs a language extension installed (provides symbols/diagnostics). |

---

## Scope

Real today: chat panel with streaming thinking + step narration, proposed-edit diffs shown
before they're applied, reject-with-feedback, Stop/interrupt, image attachments, LSP context, and
terminal commands. Not yet: applying edits through VS Code's own edit API (Hera writes to disk via
its tools, shown as a diff), built-in vision (set `hera.visionUrl` to a vision endpoint), and a
Marketplace listing.
