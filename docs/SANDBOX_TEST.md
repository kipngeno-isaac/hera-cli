# Hera `run_bash` Sandbox — End-to-End Test

This documents the end-to-end verification of the Hera CLI's `run_bash` sandbox and permission
model. The sandbox confines what shell commands the agent runs can do; see the sandbox/allowlist
sections of [`../README.md`](../README.md) for usage.

- **Date:** 2026-06-12
- **Host:** this server (`kra-gpu`, Ubuntu 22.04)
- **Sandbox backend:** bubblewrap `0.6.1` (`/usr/bin/bwrap`)
- **Hera config:** `HERA_SANDBOX=auto` → resolves to `bwrap` ("fs confined to cwd, network on")

---

## What the sandbox guarantees

When `run_bash` executes a command, it is wrapped so that:

- the whole filesystem is **read-only**, except the current working directory (and a private
  `/tmp`), which are writable;
- **network is enabled** by default (`HERA_SANDBOX_NET=0` disables it again);
- the process runs in its own PID namespace and dies with the parent.

On hosts without bubblewrap, Hera falls back to `unshare` (network/PID isolation, no filesystem
confinement) or `none`; the active level is shown in the banner and via `/sandbox`.

---

## Test 1 — confinement enforced during a real agent run

Hera was asked (in a scratch dir `~/hera_e2e`) to **write and run** a `probe.py` that attempts a
write inside cwd, a write outside cwd, and a network call, then report each result. The agent did
the full reason→act loop (`write_file` → `run_bash`) with the sandbox active. This probe was run
with `HERA_SANDBOX_NET=0` to verify that the stricter no-network mode still works. The script's
own output, as run by the agent:

| Probe (executed inside the sandbox) | Result |
|---|---|
| Write `./out.txt` (inside cwd) | ✅ wrote `"inside"` |
| Write `/home/ubuntu/hera_escape.txt` (outside cwd) | 🚫 `BLOCKED: [Errno 30] Read-only file system` |
| `urllib.request.urlopen("https://example.com", timeout=3)` | 🚫 `NET_BLOCKED` (DNS resolution failed) |

**Out-of-band verification:**
- `~/hera_e2e/out.txt` existed afterward with contents `inside`.
- `~/hera_escape.txt` was **never created** — the escape attempt was blocked, not just reported.

## Test 2 — network toggle

Same `bwrap` backend, varying only `HERA_SANDBOX_NET`:

| Mode | `curl https://example.com` |
|---|---|
| default (`net on`) | `http=200` (reachable) |
| `HERA_SANDBOX_NET=0` | exit `6` (blocked) |

Network is on by default and cleanly disabled when explicit isolation is required.

## Test 3 — `/tmp`-rooted working directory

Earlier a bug caused `bwrap: Can't chdir` when the working directory was itself under `/tmp`
(the private `--tmpfs /tmp` masked the cwd bind). Fixed by laying down the tmpfs **before**
binding cwd. Re-verified for both a `/tmp`-rooted cwd and a normal `$HOME`-rooted cwd: writes
confined to cwd, outside writes blocked, no escape.

---

## Permission model (tested separately)

- **Allowlist:** `HERA_ALLOW` / `.heraallow` / `/allow <pattern>` / `[a]`/`[p]` at a prompt
  auto-approve matching `run_bash` commands. Verified an allowlisted command runs with no prompt
  (`↳ auto-approved (allowlist)`).
- **Denylist:** built-in dangerous patterns (`rm -rf /`, `sudo …`, `mkfs`, `curl … | sh`,
  fork-bombs, …) always force a prompt even if an allow pattern would match; deny wins.
- **Approval gate:** denied tool calls return "user declined" to the model (which adapts);
  read-only tools (`list_dir`, `read_file`, `glob`, `search`) never prompt.

---

## Conclusion

The `run_bash` sandbox is enforced on the agent's own tool calls (not just in isolated tests):
writes are confined to the working directory, the rest of the filesystem is read-only, network is
available unless explicitly disabled, layered on top of the approval gate and allowlist/denylist.
