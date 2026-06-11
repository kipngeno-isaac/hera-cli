# Accessing the model from the Web (Open WebUI)

**Open WebUI** is the browser chat interface for the Qwen3.6-35B-A3B model. It's a full
multi-user system: every user logs in with their **email**, has their **own private chat
history**, and an admin decides who's allowed in. Your account here is the **same identity** you
use for the Hera CLI and the VS Code extension.

> See also [`ACCESS_CLI.md`](ACCESS_CLI.md) (terminal) and [`ACCESS_VSCODE.md`](ACCESS_VSCODE.md).
> Replace `<HOST>` with the server address your admin gives you.

---

## 1. Get access

Anyone can **sign up**, but a new account is **`pending`** until an admin approves it — you
can't chat or use the API until then.

1. Go to **`http://<HOST>:3000`** → **Sign up** with your name, email, and a password.
2. You'll see a "pending activation" screen. Ask the admin to approve you.
3. Once approved, **sign in** and start chatting.

> Prefer the admin to create your account directly? They still can (Admin Panel → Users) — either
> way you end up with the same email login.

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

- **Approve users:** *Admin Panel → Users*. Anyone can self-register (`ENABLE_SIGNUP=true`), but
  new accounts default to **`pending`** (`DEFAULT_USER_ROLE=pending`) and can't do anything until
  you set their role to `user`. You can also add users manually here. Only `user`/`admin` roles
  can use the API (the identity proxy rejects `pending`).
- **API keys feature:** enabled, and the **`user` role permission for API keys is on** (*Admin
  Panel → Settings → Users → Permissions → Features → API Keys*) so approved users can actually
  mint the personal keys used by the CLI/extension. (Without that permission the API Keys section
  is hidden for non-admins even though the feature is enabled.)
- **Remove access:** set a user's role to `pending` or delete them — that immediately blocks both
  their web login and their CLI/extension access (the proxy revalidates against this account).
- **Model wiring:** Open WebUI talks to the inference server via `OPENAI_API_BASE_URL` /
  `OPENAI_API_KEY` (configured in `docker-compose.yml`); users never see that shared key.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Signed up but can't chat | Expected — your account is `pending`. Ask the admin to approve you (set role to `user`). |
| No model in the selector | The inference server may be reloading; try again shortly, or tell the admin to check `qwen-server`. |
| "Account is not approved" from the CLI | Your role is `pending`; ask the admin to set it to `user`. |
| Lost your API key | Recreate it in Settings → Account → API keys (the old one stops working). |
