# Accessing the model from the Web (Open WebUI)

**Open WebUI** is the browser chat interface for the Qwen3.6-35B-A3B model. It's a full
multi-user system: every user logs in with their **email**, has their **own private chat
history**, and an admin decides who's allowed in. Your account here is the **same identity** you
use for the Hera CLI and the VS Code extension — one login, three surfaces.

> See also [`ACCESS_CLI.md`](ACCESS_CLI.md) (terminal) and [`ACCESS_VSCODE.md`](ACCESS_VSCODE.md).

---

## 0. What you need first

| Thing | Value |
|---|---|
| **`<HOST>`** | The server address your admin gives you (an IP like `203.0.113.5` or a hostname). Substitute it everywhere `<HOST>` appears below. |
| Web UI URL | `http://<HOST>:3000` |
| A browser | Any modern one. Nothing to install for web chat. |

The three user-facing ports in this stack: **`:3000`** Open WebUI (web chat + key management),
**`:8090`** the identity proxy (what the CLI/extension talk to), **`:8081`** the CLI installer
download host. As a web user you only ever touch `:3000`.

---

## 1. Get access

Anyone can **sign up**, but a new account is **`pending`** until an admin approves it — you
can't chat or use the API until then.

1. Go to **`http://<HOST>:3000`** → **Sign up** with your name, email, and a password.
2. You'll see a "pending activation" screen. Ask the admin to approve you.
3. Once approved, **sign in** and start chatting.

> Prefer the admin to create your account directly? They still can (Admin Panel → Users → **+**)
> — either way you end up with the same email login.

Your chats are private to your account; nothing is shared with other users.

> **Setting up the very first account?** See [§5 First-time deployment](#5-first-time-deployment-bootstrapping-the-first-admin)
> — the **first** account to sign up automatically becomes the **admin**, and there's no one to
> approve it (it's approved on creation).

---

## 2. Using the chat

1. Pick the model **`qwen3.6-35b-a3b`** from the model selector at the top.
2. Type your message and send. Conversations are saved in the left sidebar (yours only).
3. It's a **reasoning model** — it may "think" before answering; the final answer streams in.

That's it for normal chat use. Standard Open WebUI features (rename/organize chats, regenerate,
edit, upload files for RAG, etc.) all work and stay within your account.

### Shared skills

This stack also has a **Shared Skills** layer so the same domain/workflow guidance can apply in
the web chat, Hera CLI, and the VS Code extension.

- Skills are activated automatically when your message matches a skill's trigger phrases.
- You can force one on explicitly by writing `@skill:<id>` or `/skill <id>` in your message.
- Admins can add or update skills in the repo under [`shared-skills/skills/`](shared-skills/skills/).
- Admins can see the current live catalog in **Admin Panel → Functions → Shared Skills** via the
  function's visible `catalog` field.
- Skills can be disabled globally in [`shared-skills/config.json`](shared-skills/config.json) with
  the `disabled_skills` list, or by setting `enabled: false` in a skill file.

Current built-in shared skills include:
- `live-research`
- `kenya-tax`
- `software-delivery`
- `world-time`

### Live web search (current info)

The model can pull **live data from the internet** so it isn't limited to what it was trained on.
Open WebUI still has the **🌐 globe / Web Search** toggle, but on this stack it also has a global
**Auto Web Search** filter: prompts that clearly need current or external facts automatically turn
web search on before the model answers. That includes queries like **"what's the time right now in
Nairobi"**, weather, prices, news, "latest version", and similar time-sensitive requests. Open
WebUI then searches the web (DuckDuckGo), fetches the top pages, feeds their full text to the
model, and the answer comes back **with inline citations** you can click to open the source.

For **exact time/date/day** questions, the browser chat now prefers the fast `world-time` shared
skill instead of DDG. That path asks the internal skill API for the current time and injects the
answer directly, so it avoids the slow / empty-source failure mode you saw before.

- The 🌐 toggle still works as a **manual override**: use it to force browsing on a prompt that the
  heuristic didn't catch, or to make the intent explicit.
- Use it for anything **current or factual**: recent releases, today's news, library/API docs,
  "what's the latest version of X", error messages, etc.
- Ordinary chat/coding still stays fast because the auto filter only triggers on prompts that look
  time-sensitive or fact-dependent; it does **not** browse every message by default.
- The model reads the fetched pages and **synthesizes** an answer; the citations show what it used.

> Admins: this is enabled stack-wide (`ENABLE_WEB_SEARCH`, engine `duckduckgo`, no API key). See
> [§4](#4-for-the-admin) to change the engine or result count.

> **Don't see the model in the selector?** You're probably still `pending` — ask the admin to
> approve you. If you *are* approved and it's still missing, the inference server may be
> reloading; wait a moment and refresh (admins: see [§4](#4-for-the-admin)).

---

## 3. Create an API key (to use the CLI / VS Code as the same user)

Your API key is what links your web account to the terminal CLI and the editor extension. It is
your **identity** there — sessions are labelled by the account it belongs to.

1. Click your **name / avatar** (bottom-left) → **Settings**.
2. **Account → API keys → Create new key** (`+ Create new key`).
3. Copy the `sk-…` key somewhere safe — this is the only time it's shown in full.

On this stack the API-keys panel gives you three controls:

- **Create** — mint a new key.
- **Regenerate** — rotate the key (the old one stops working immediately).
- **Delete** (🗑) — revoke the key entirely. *This Delete button is a customization of this
  deployment* (upstream Open WebUI ships the delete endpoint but no button).

Use the key as `HERA_API_KEY` for the [CLI](ACCESS_CLI.md) / [VS Code](ACCESS_VSCODE.md), pointed
at the identity proxy `http://<HOST>:8090/v1`. Same login, same identity, isolated context — on
web **and** terminal.

> Keep the key secret (it acts as you). You can revoke/recreate it any time in the same screen.
> If you lose it, just **Create** a new one — the old one keeps working until you delete/regenerate it.

---

## 4. For the admin

You decide who gets in and manage accounts from the **Admin Panel** (your avatar → **Admin
Panel**).

- **Approve users:** *Admin Panel → Users*. Anyone can self-register (`ENABLE_SIGNUP=true`), but
  new accounts default to **`pending`** (`DEFAULT_USER_ROLE=pending`) and can't do anything until
  you set their **role** to `user`. You can also add users manually here (**+**). Only
  `user`/`admin` roles can use the API (the identity proxy rejects `pending`).
- **API keys feature:** enabled, and the **`user` role permission for API keys is on** (*Admin
  Panel → Settings → Users → Permissions → Features → API Keys*) so approved users can actually
  mint the personal keys used by the CLI/extension. (Without that permission the API Keys section
  is **hidden** for non-admins even though the global feature is enabled — two separate switches.)
- **Model visibility:** every approved user sees the shared `qwen3.6-35b-a3b` model because
  `BYPASS_MODEL_ACCESS_CONTROL=true` is set. Without it, a raw connection model (no Workspace →
  Models entry) is **admin-only** and regular users get an empty list / "Model not found". (Granular
  alternative: publish the model via *Admin Panel → Settings → Models → set access to **Public***.)
- **Remove access:** set a user's role back to `pending` or delete them — that immediately blocks
  both their web login and their CLI/extension access (the proxy revalidates against this account
  on every request, with a ~60 s cache).
- **Model wiring:** Open WebUI talks to the inference server via `OPENAI_API_BASE_URL` /
  `OPENAI_API_KEY` (set in `docker-compose.yml`); users never see that shared key.
- **Web search:** enabled stack-wide so users can browse from chat (the 🌐 toggle), and the custom
  OWUI image seeds a global **Auto Web Search** filter into the DB on startup so current/factual
  prompts can browse without the user toggling it manually. Core search is configured in
  `docker-compose.yml`: `ENABLE_WEB_SEARCH=true`, `WEB_SEARCH_ENGINE=duckduckgo` (needs no API
  key), `WEB_SEARCH_RESULT_COUNT=5`, and `BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=true` (the
  fetched pages are injected straight into the model's 32k context — no embedding model to host,
  and the model sees the full sources). The per-role **Web Search** permission
  (`USER_PERMISSIONS_FEATURES_WEB_SEARCH`) is on. To change the engine, result count, or switch to
  an API-key search provider on a running instance, use *Admin Panel → Settings → Web Search*
  (PersistentConfig — editing the env var afterwards won't take; see the gotcha below). To tune or
  disable the heuristic itself, use *Admin Panel → Functions → Auto Web Search*.
- **Shared skills:** the custom OWUI image also seeds a global **Shared Skills** filter, and the
  `shared-skills/` repo folder is mounted into the container read-only. Update skill files in
  [`shared-skills/skills/`](shared-skills/skills/) and restart `open-webui` to refresh what the
  browser chat injects. Users can force a skill with `@skill:<id>` or `/skill <id>`. The function
  also exposes a visible `catalog` field in Admin → Functions so admins can see the current live
  skill list and status.

> **⚠ PersistentConfig gotcha (important).** Open WebUI persists most of these settings
> (`ENABLE_SIGNUP`, `DEFAULT_USER_ROLE`, the API-key permission, model access, etc.) into its
> **database** on first boot. After that, **changing the env var in `docker-compose.yml` has no
> effect** — the DB value wins. To change a setting on a running instance, change it in the
> **Admin Panel UI** (or wipe the `open-webui-data` volume to re-seed from env, which also deletes
> all users/chats). So: set env vars *before* the first boot; tweak everything afterwards in the UI.

---

## 5. First-time deployment (bootstrapping the first admin)

If you're standing the stack up from scratch, there's no admin yet to approve anyone — so the
order matters:

1. Bring the stack up (`docker compose up -d`) and open **`http://<HOST>:3000`**.
2. **Sign up.** The **first account ever created is automatically made `admin`** and is approved
   on creation (this overrides `DEFAULT_USER_ROLE=pending`, which only applies to *subsequent*
   signups). Use a real email/password you'll keep — this is your admin login.
3. As that admin, go to **Admin Panel → Settings → Users → Permissions → Features** and confirm
   **API Keys** is on for the `user` role (see [§4](#4-for-the-admin)).
4. Everyone else now signs up normally, lands as `pending`, and you approve them in
   **Admin Panel → Users**.

After that, hand each approved user [`ACCESS_WEB.md`](ACCESS_WEB.md) (mint a key) plus
[`ACCESS_CLI.md`](ACCESS_CLI.md) / [`ACCESS_VSCODE.md`](ACCESS_VSCODE.md), and tell them the
`<HOST>` address.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Signed up but can't chat | Expected — your account is `pending`. Ask the admin to approve you (set role to `user`). |
| No model in the selector | You're `pending` (most common), or the inference server is reloading — try again shortly, or tell the admin to check `qwen-server`. |
| "Account is not approved" from the CLI | Your role is `pending`; ask the admin to set it to `user`. |
| API Keys section missing in Settings | The per-role API-key **permission** is off — admin enables it under *Settings → Users → Permissions → Features → API Keys*. |
| Lost your API key | Create a new one in Settings → Account → API keys (the old one keeps working until you delete/regenerate it). |
| Changed an env var but nothing changed | PersistentConfig — change it in the **Admin Panel UI** instead (see the gotcha in [§4](#4-for-the-admin)). |
| Can't reach `http://<HOST>:3000` at all | Wrong `<HOST>`, the stack is down, or a firewall blocks `:3000`. Ask the admin. |
