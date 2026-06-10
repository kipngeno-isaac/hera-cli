# Accessing Hera in VS Code / Cursor

The **Hera extension** brings the agent into your editor: a native **chat panel** (streaming
answers, tool cards, inline approval buttons), **in-editor diffs** for every edit Hera makes,
and **LSP-driven context** (send the file's symbols + diagnostics with your question). It drives
the same `hera` CLI under the hood — one agent, just a graphical surface.

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
3. **`bubblewrap`** (optional, recommended) for full shell sandboxing.

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
vsce package                       # produces hera-cli-0.1.0.vsix
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
  "hera.extraEnv": { "HERA_USER": "you@example.com" },
  "hera.showDiffs": true
}
```

- `hera.serverUrl` → the **identity proxy** (`:8090/v1`).
- `hera.apiKey` → your personal Open WebUI key (or leave blank to use your shell's `HERA_API_KEY`).
- `hera.extraEnv` → any extra env (e.g. `HERA_USER`, `HERA_YOLO`, `HERA_EMBED_URL`).

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
- The model's **thinking** appears in a collapsible block; the **answer** renders as markdown.
- Each tool call shows as a card (`◆ tool → result`); when Hera **edits a file**, a native
  **diff editor** opens (before ↔ after) so you can review the change. Toggle with `hera.showDiffs`.
- When a write/edit/shell command needs approval, **buttons** appear inline
  (*Approve once / Always / Deny*).
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

Real today: chat panel, in-editor diffs, LSP context, terminal commands. Not yet: applying edits
through VS Code's own edit API (Hera writes to disk via its tools, shown as a diff) and a
Marketplace listing.
