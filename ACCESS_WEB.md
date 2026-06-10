# Accessing the model from the Web (Open WebUI)

**Open WebUI** is the browser chat interface for the Qwen3.6-35B-A3B model. It's a full
multi-user system: every user logs in with their **email**, has their **own private chat
history**, and an admin decides who's allowed in. Your account here is the **same identity** you
use for the Hera CLI and the VS Code extension.

> See also [`ACCESS_CLI.md`](ACCESS_CLI.md) (terminal) and [`ACCESS_VSCODE.md`](ACCESS_VSCODE.md).
> Replace `<HOST>` with the server address your admin gives you.

---

## 1. Get access

Open signup is **disabled** — an admin must create or approve your account.

1. Ask the admin to add you (they set your email + an initial password, or approve a pending
   request).
2. Go to **`http://<HOST>:3000`** and **sign in** with your email.

Your chats are private to your account; nothing is shared with other users.

---

## 2. Using the chat

1. Pick the model **`qwen3.6-35b-a3b`** from the model selector at the top.
2. Type your message and send. Conversations are saved in the left sidebar (yours only).
3. It's a **reasoning model** — it may "think" before answering; the final answer streams in.

That's it for normal chat use. Standard Open WebUI features (rename/organize chats, regenerate,
edit, etc.) all work and stay within your account.

---

## 3. Create an API key (to use the CLI / VS Code as the same user)

Your API key is what links your web account to the terminal CLI and the editor extension.

1. Click your name (bottom-left) → **Settings**.
2. **Account → API keys → Create new key**.
3. Copy the `sk-…` key.

Use it as `HERA_API_KEY` for the [CLI](ACCESS_CLI.md) / [VS Code](ACCESS_VSCODE.md), pointed at
the identity proxy `http://<HOST>:8090/v1`. Same login, same identity, isolated context — on web
**and** terminal.

> Keep the key secret (it acts as you). You can revoke/recreate it any time in the same screen.

---

## 4. For the admin

You decide who gets in and manage accounts from the **Admin Panel** (your avatar → **Admin
Panel**).

- **Add / approve users:** *Admin Panel → Users*. New people can't self-register
  (`ENABLE_SIGNUP=false`); create them here, or approve anyone whose role is `pending` by
  setting it to `user`. Only `user`/`admin` roles can use the API (the identity proxy rejects
  `pending`).
- **API keys feature:** already enabled (*Admin Panel → Settings → General → API keys*). This is
  what lets approved users mint the personal keys used by the CLI/extension.
- **Remove access:** set a user's role to `pending` or delete them — that immediately blocks both
  their web login and their CLI/extension access (the proxy revalidates against this account).
- **Model wiring:** Open WebUI talks to the inference server via `OPENAI_API_BASE_URL` /
  `OPENAI_API_KEY` (configured in `docker-compose.yml`); users never see that shared key.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Can't sign up | Expected — signup is disabled. Ask the admin to create your account. |
| No model in the selector | The inference server may be reloading; try again shortly, or tell the admin to check `qwen-server`. |
| "Account is not approved" from the CLI | Your role is `pending`; ask the admin to set it to `user`. |
| Lost your API key | Recreate it in Settings → Account → API keys (the old one stops working). |
