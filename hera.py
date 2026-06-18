#!/usr/bin/env python3
"""
Hera SYM_EMDASH an agentic coding CLI for the Qwen3.6-35B-A3B model.

Hera runs the model in a reasonSYM_ARROW_Ract loop with real tools: it can list
directories, find files, search code, read/write/edit files, and run shell
commands in the directory you launch it from. It asks for approval before
editing files or running commands, streams the model's reasoning, and tracks
token usage.

Required:       HERA_API_URL  (the endpoint, e.g. http://<host>:8080/v1 SYM_EMDASH no host is
                              baked in, so this file can live in a public repo)
                HERA_API_KEY  (bearer key the server enforces)
Optional:       HERA_MODEL          (default qwen3.6-35b-a3b)
                HERA_NAME           (assistant display name; default Hera)
                HERA_YOLO=1         auto-approve every tool call (no prompts)
                HERA_MAX_STEPS      max tool round-trips per message (default: unlimited)
                HERA_HIDE_REASONING=1   don't stream the model's thinking
                HERA_NO_SUGGESTIONS=1   don't print "Next steps" tips after a task
                HERA_NO_VERIFY=1    don't auto-run/verify code after editing it
                HERA_PLAN=1         start in plan mode (investigate & propose,
                                    no edits until you /plan to approve)
                HERA_PRICE_IN / HERA_PRICE_OUT   USD per 1M tokens SYM_ARROW_R show $ cost
                HERA_CONTEXT_TOKENS / HERA_AUTO_COMPACT_AT   auto-compact history
                                    when it nears the context window (default 32000, 0.8)
                HERA_VISION_URL     vision endpoint for image attachments (the
                                    main model is text-only; without this, images
                                    are attached but not interpreted)
                HERA_VISION_MODEL   model name at HERA_VISION_URL

Config file (~/.config/hera/config.json) also supports:
    "hooks":       {"PreToolUse":[{"matcher":"run_bash","command":"SYM_ELLIPSIS"}], "PostToolUse":[SYM_ELLIPSIS], "Stop":[SYM_ELLIPSIS]}
    "permissions": {"allow":["run_bash(git *)"], "ask":["write_file"], "deny":["run_bash(rm *)"]}
    "price_in"/"price_out", "context_tokens", "auto_compact_at"
Custom slash commands live in ~/.config/hera/commands/*.md ($ARGUMENTS),
named sub-agents in ~/.config/hera/agents/*.md (optional `tools:` frontmatter).

Legacy QWEN_* variables (and LLAMA_API_KEY) are still honoured as fallbacks.
"""
import argparse
import ast
import base64
import contextlib
import difflib
import fnmatch
import glob as globmod
import hashlib
import html as ihtml
import importlib.util
import json
import math
import mimetypes
import os
import queue
import random
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import urllib.parse as _urlparse

try:
    import readline  # noqa: F401  (enables line editing in the input() fallback)
except ImportError:
    pass

try:
    import termios
    import tty
except ImportError:  # non-POSIX (e.g. Windows) SYM_EMDASH raw mode unavailable
    termios = None
    tty = None

try:
    import requests
except ImportError:
    sys.exit(
        "[error] 'requests' not found.\n"
        "Install it with:  pip install requests\n"
        "Then re-run:      hera"
    )


# ── Config ───────────────────────────────────────────────────────────────────
def _env(*names, default=""):
    """First non-empty value among the given env var names, else default."""
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


def _truthy(v):
    return v not in ("", "0", "false", "False", "no", "off")


# Persistent on-disk config so a user only ever pastes their key once. The
# installer writes the endpoint here; first run captures the key (see
# onboard()). Env vars always win over the file.
CONFIG_PATH = os.path.expanduser(_env("HERA_CONFIG", default="~/.config/hera/config.json"))


def _load_config_file():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_FILE_CFG = _load_config_file()


def _cfg(*env_names, key=None, default=""):
    """Env var (first non-empty) SYM_ARROW_R config file[key] SYM_ARROW_R default."""
    v = _env(*env_names)
    if v:
        return v
    if key and _FILE_CFG.get(key):
        return _FILE_CFG[key]
    return default


def _cfg_truthy(*env_names, key=None, default=False):
    """Env var SYM_ARROW_R config file value SYM_ARROW_R boolean default."""
    v = _env(*env_names)
    if v != "":
        return _truthy(v)
    if key:
        cfg_v = _FILE_CFG.get(key)
        if cfg_v not in (None, ""):
            if isinstance(cfg_v, bool):
                return cfg_v
            return _truthy(str(cfg_v))
    return default


def save_config(updates):
    """Merge `updates` into the on-disk config (0600). Best-effort."""
    _FILE_CFG.update({k: v for k, v in updates.items() if v})
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_FILE_CFG, f, indent=2)
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


VERSION = "0.8.38"   # bump on every released change; mirrored in cli/VERSION
NAME    = _env("HERA_NAME", default="Hera")
# No server host is baked into the source (so this repo can be public, revealing
# neither key nor host). Each user supplies the endpoint + key once — via env
# vars, the installer-written config file, or the first-run paste prompt.
API_URL = _cfg("HERA_API_URL", "QWEN_API_URL", key="api_url", default="").rstrip("/")
MODEL   = _env("HERA_MODEL",   "QWEN_MODEL",   default="qwen3.6-35b-a3b")
API_KEY = _cfg("HERA_API_KEY", "QWEN_API_KEY", "LLAMA_API_KEY", key="api_key", default="")

# Provider backend. "openai" (default) talks to any OpenAI-compatible endpoint
# (llama.cpp, vLLM, OpenAI, etc.). "anthropic"/"bedrock"/"vertex" use the native
# Anthropic Messages API (Bedrock & Vertex are base-URL + auth variants of it).
PROVIDER = (_cfg("HERA_PROVIDER", key="provider", default="openai") or "openai").lower()
ANTHROPIC_BASE = _cfg("HERA_ANTHROPIC_BASE", key="anthropic_base",
                      default="https://api.anthropic.com").rstrip("/")
ANTHROPIC_VERSION = _env("HERA_ANTHROPIC_VERSION", default="2023-06-01")
MAX_OUTPUT_TOKENS = int(_env("HERA_MAX_OUTPUT_TOKENS", default="32768") or 32768)
# Vim keybindings in the prompt editor (toggle with /vim or HERA_VIM=1).
VIM_MODE = _truthy(_env("HERA_VIM"))
# Vision: the main model is text-only. If a vision-capable OpenAI-compatible
# endpoint is configured, turns that carry an attached image are routed to it;
# otherwise images are attached but flagged as not-interpreted (see stream_turn).
VISION_URL   = _cfg("HERA_VISION_URL", key="vision_url", default="").rstrip("/")
VISION_MODEL = _env("HERA_VISION_MODEL", default="")
IMAGE_EXTS   = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
YOLO    = _truthy(_env("HERA_YOLO", "QWEN_YOLO"))
MAX_STEPS      = int(_env("HERA_MAX_STEPS", "QWEN_MAX_STEPS", default="0"))  # 0 = unlimited
HIDE_REASONING = _truthy(_env("HERA_HIDE_REASONING", "QWEN_HIDE_REASONING"))

# Output style (Claude-Code-style): shapes how answers are written. Built-ins
# below; custom styles live in ~/.config/hera/output-styles/<name>.md.
OUTPUT_STYLE = _cfg("HERA_OUTPUT_STYLE", key="output_style", default="default") or "default"
_OUTPUT_STYLES = {
    "default":     "",
    "concise":     ("Answer as briefly as possible: short, direct, minimal prose. Prefer a one-line "
                    "answer or a tight list. No preamble, no recap unless asked."),
    "explanatory": ("Explain your reasoning as you work: briefly note why you chose an approach and "
                    "call out trade-offs and non-obvious decisions, so the user learns from each step."),
    "learning":    ("Teach as you go: explain concepts the user may not know, and occasionally leave a "
                    "small, clearly-marked 'TODO(you):' for the user to implement, then review it."),
}

# Thinking budget: off | normal | hard. Maps to the model's enable_thinking
# chat-template kwarg (and a deeper-reasoning nudge for 'hard').
THINK_LEVEL = (_cfg("HERA_THINK", key="think", default="normal") or "normal").lower()

# Optional status line: a command whose stdout is shown above the prompt each
# turn (session JSON on stdin), like Claude Code's statusLine.
STATUSLINE_CMD = _cfg("HERA_STATUSLINE", key="statusline", default="")

MAX_TOOL_OUTPUT = int(_env("HERA_MAX_TOOL_OUTPUT", default="0"))  # 0 = unlimited
MAX_READ_BYTES  = 256_000  # cap read_file size
MAX_IMAGE_BYTES = 10_000_000  # cap an attached image before base64 (SYM_APPROX13MB encoded)

# Web access: the model can search/fetch the live web when it lacks info.
WEB_ENABLED = not _truthy(_env("HERA_NO_WEB"))      # on by default; HERA_NO_WEB=1 disables
WEB_TIMEOUT = int(_env("HERA_WEB_TIMEOUT", default="60"))
_WEB_UA = f"Mozilla/5.0 (X11; Linux x86_64) HeraCLI/{VERSION}"
# Web search provider: duckduckgo (default, keyless HTML scrape) or a higher-
# quality API — tavily / brave / searxng — via HERA_SEARCH_KEY / HERA_SEARCH_URL.
SEARCH_PROVIDER = (_cfg("HERA_SEARCH_PROVIDER", key="search_provider",
                        default="duckduckgo") or "duckduckgo").lower()
SEARCH_KEY = _cfg("HERA_SEARCH_KEY", key="search_key", default="")
SEARCH_URL = _cfg("HERA_SEARCH_URL", key="search_url", default="").rstrip("/")

# Offer to install a missing program rather than just failing a run_bash call.
AUTO_INSTALL = not _truthy(_env("HERA_NO_AUTOINSTALL"))
# Commands whose package name differs from the binary name.
_PKG_MAP = {"rg": "ripgrep", "fd": "fd-find", "http": "httpie", "pip": "python3-pip",
            "pip3": "python3-pip", "convert": "imagemagick", "aws": "awscli"}

# Sandboxing for run_bash. Modes: auto | bwrap | unshare | none
SANDBOX_MODE = _env("HERA_SANDBOX", default="auto").lower()
# Keep shell networking available by default; users can still disable it with
# HERA_SANDBOX_NET=0 or "sandbox_net": false in config.json.
SANDBOX_NET  = _cfg_truthy("HERA_SANDBOX_NET", key="sandbox_net", default=True)

# Running token usage for the whole session.
SESSION = {"prompt": 0, "completion": 0, "total": 0, "requests": 0}
# The server-reported total tokens of the last request — used as ground truth for
# auto-compaction (more accurate than estimating from characters).
_LAST_PROMPT_TOKENS = 0

# ── Claude-parity feature config ──────────────────────────────────────────────
# Cost estimate: USD price per 1M tokens. 0 (the default) hides the $ display.
PRICE_IN  = float(_cfg("HERA_PRICE_IN",  key="price_in",  default="0") or 0)
PRICE_OUT = float(_cfg("HERA_PRICE_OUT", key="price_out", default="0") or 0)

# Auto-compaction: when the estimated prompt size crosses AUTO_COMPACT_AT * the
# context window, the history is summarized before the next turn. 0 disables.
CONTEXT_TOKENS  = int(_cfg("HERA_CONTEXT_TOKENS", key="context_tokens", default="131072") or 0)
AUTO_COMPACT_AT = float(_cfg("HERA_AUTO_COMPACT_AT", key="auto_compact_at", default="0.8") or 0)

# User hooks: config["hooks"] = {"PreToolUse":[{"matcher":"run_bash","command":"..."}], …}.
# Events: PreToolUse, PostToolUse, Stop, UserPromptSubmit, SessionStart, SessionEnd,
# PreCompact, SubagentStop, Notification. A PreToolUse/UserPromptSubmit hook exiting
# non-zero blocks; UserPromptSubmit/SessionStart stdout is injected as extra context.
HOOKS = _FILE_CFG.get("hooks") if isinstance(_FILE_CFG.get("hooks"), dict) else {}
_HOOK_CONTEXT = ""   # stdout from the last UserPromptSubmit/SessionStart hooks

# Custom keybindings: ~/.config/hera/keybindings.json maps "ctrl+r" → "/review" etc.
# Supported: ctrl+[a-z] (byte 0x01-0x1a), alt+[a-z] (ESC+char).
def _load_keybindings():
    _cfg_dir = os.path.dirname(CONFIG_PATH) or os.path.expanduser("~/.config/hera")
    p = os.path.join(_cfg_dir, "keybindings.json")
    if not os.path.isfile(p):
        return {}
    try:
        raw = json.loads(open(p, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return {}
    ctrl_map, alt_map = {}, {}
    for key_str, cmd in raw.items():
        k = key_str.lower().strip()
        if k.startswith("ctrl+") and len(k) == 6 and k[5].isalpha():
            byte = ord(k[5]) - ord('a') + 1  # ctrl+a=0x01 SYM_ELLIPSIS ctrl+z=0x1a
            ctrl_map[byte] = cmd
        elif k.startswith("alt+") and len(k) == 5 and k[4].isalpha():
            alt_map[k[4]] = cmd
    return {"ctrl": ctrl_map, "alt": alt_map}

KEYBINDINGS = _load_keybindings()
_KB_CTRL = KEYBINDINGS.get("ctrl", {})
_KB_ALT  = KEYBINDINGS.get("alt", {})

# Fine-grained permissions: config["permissions"] = {"allow":[...],"ask":[...],"deny":[...]}.
# Each entry is "tool" or "tool(<glob>)" — e.g. "run_bash(git *)", "edit_file(src/**)".
_PERMS = _FILE_CFG.get("permissions") if isinstance(_FILE_CFG.get("permissions"), dict) else {}

# Enterprise managed policy (read-only, admin-controlled): a JSON file that
# OVERRIDES user config and cannot be loosened by the user. Keys:
#   permissions {deny,ask,allow}  — managed rules win over everything
#   disable_bypass: true          — forces YOLO/bypass off
#   max_auto_mode: read|edit|all  — caps the per-project auto-approve level
#   deny_patterns: ["rm -rf *"]   — extra run_bash deny patterns
def _load_managed_policy():
    path = _env("HERA_MANAGED_POLICY") or "/etc/hera/managed-policy.json"
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}


MANAGED_POLICY = _load_managed_policy()
_MANAGED_PERMS = MANAGED_POLICY.get("permissions") if isinstance(
    MANAGED_POLICY.get("permissions"), dict) else {}

# Telemetry (OpenTelemetry-style): emit usage/events to an OTLP-ish HTTP endpoint
# and/or a local JSONL file. Off unless a sink is configured.
TELEMETRY_LOG = _cfg("HERA_TELEMETRY_LOG", key="telemetry_log", default="")
OTEL_ENDPOINT = _cfg("HERA_OTEL_ENDPOINT", key="otel_endpoint", default="").rstrip("/")
TELEMETRY_ON = bool(TELEMETRY_LOG or OTEL_ENDPOINT)

# Enterprise policy can force bypass/YOLO off; the user can't re-enable it.
if MANAGED_POLICY.get("disable_bypass"):
    YOLO = False


def _managed_cap_auto_mode(mode):
    """Clamp an auto-approve level to the managed policy's ceiling, if any."""
    cap = MANAGED_POLICY.get("max_auto_mode")
    order = {"read": 0, "edit": 1, "all": 2}
    if cap in order and order.get(mode, 0) > order[cap]:
        return cap
    return mode


_TELEMETRY_LOCK = threading.Lock()


def _otlp_log(event, attrs):
    """Minimal OTLP/HTTP logs payload for one event."""
    return {"resourceLogs": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "hera"}},
            {"key": "service.version", "value": {"stringValue": VERSION}}]},
        "scopeLogs": [{"logRecords": [{
            "timeUnixNano": int(time.time() * 1e9),
            "body": {"stringValue": event},
            "attributes": [{"key": k, "value": {"stringValue": str(v)}}
                           for k, v in attrs.items()]}]}]}]}


def _emit_telemetry(event, **attrs):
    """Emit a usage/event record to the configured telemetry sinks (JSONL file
    and/or an OTLP/HTTP endpoint). Best-effort; never raises, never blocks long."""
    if not TELEMETRY_ON:
        return
    rec = {"ts": time.time(), "event": event,
           "session": CURRENT_SESSION.get("id"), "model": MODEL,
           "provider": PROVIDER, **attrs}
    if TELEMETRY_LOG:
        try:
            with _TELEMETRY_LOCK, open(os.path.expanduser(TELEMETRY_LOG), "a",
                                       encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    if OTEL_ENDPOINT:
        try:
            requests.post(f"{OTEL_ENDPOINT}/v1/logs", json=_otlp_log(event, rec),
                          timeout=3)
        except Exception:  # noqa: BLE001 SYM_EMDASH telemetry must never break a turn
            pass


def _otlp_metric(name, value, attrs, monotonic=True):
    """Minimal OTLP/HTTP metrics payload for one cumulative-sum data point."""
    point = {
        "asDouble": float(value),
        "timeUnixNano": int(time.time() * 1e9),
        "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in attrs.items()],
    }
    return {"resourceMetrics": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "hera"}},
            {"key": "service.version", "value": {"stringValue": VERSION}}]},
        "scopeMetrics": [{"metrics": [{
            "name": name,
            "sum": {"dataPoints": [point], "aggregationTemporality": 2,
                    "isMonotonic": monotonic}}]}]}]}


def _emit_metric(name, value, **attrs):
    """Emit a counter metric to the telemetry sinks (JSONL and/or OTLP metrics).
    The metrics signal complements the event logs from _emit_telemetry."""
    if not TELEMETRY_ON:
        return
    rec = {"ts": time.time(), "metric": name, "value": value,
           "session": CURRENT_SESSION.get("id"), "model": MODEL, **attrs}
    if TELEMETRY_LOG:
        try:
            with _TELEMETRY_LOCK, open(os.path.expanduser(TELEMETRY_LOG), "a",
                                       encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    if OTEL_ENDPOINT:
        try:
            requests.post(f"{OTEL_ENDPOINT}/v1/metrics",
                          json=_otlp_metric(name, value, {**attrs, "model": MODEL}), timeout=3)
        except Exception:  # noqa: BLE001
            pass

# Plan mode: read-only investigation; mutating tools are blocked until the user
# approves the plan (toggle with /plan, or a {"type":"plan"} message in --serve).
PLAN_MODE = _truthy(_env("HERA_PLAN"))

# Auto mode (Claude-Code-style permission modes), remembered PER PROJECT:
#   read  — only read-only tools auto-run; writes/commands still prompt   [default]
#   edit  — also auto-approve file writes/edits (shell commands still prompt)
#   all   — auto-approve every tool (deny rules, plan mode and hooks still win)
# Set with /auto (or HERA_AUTO_MODE / a {"type":"auto"} serve message); stop any
# time with /auto off (→ read). Saved under config["auto_modes"][<project path>].
_AUTO_LEVELS = ("read", "edit", "all")
_EDIT_TOOLS = {"write_file", "edit_file", "multi_edit", "notebook_edit"}


def _project_key():
    return os.path.abspath(os.getcwd())


def _load_auto_mode():
    env = _env("HERA_AUTO_MODE").lower()
    if env in _AUTO_LEVELS:
        return _managed_cap_auto_mode(env)
    modes = _FILE_CFG.get("auto_modes")
    if isinstance(modes, dict) and modes.get(_project_key()) in _AUTO_LEVELS:
        return _managed_cap_auto_mode(modes[_project_key()])
    return "read"


def _save_auto_mode(mode):
    modes = _FILE_CFG.get("auto_modes")
    if not isinstance(modes, dict):
        modes = {}
    modes[_project_key()] = mode
    save_config({"auto_modes": modes})


AUTO_MODE = _load_auto_mode()

# Auto-verify: after a turn that wrote/edited code but didn't run anything, Hera
# nudges itself once to run it (tests/build/execute) and fix failures — the
# "verify your work" loop Claude Code/Codex do. Disable with HERA_NO_VERIFY=1.
AUTO_VERIFY = not _cfg_truthy("HERA_NO_VERIFY", key="no_verify", default=False)
_CODE_EXTS = (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
              ".c", ".h", ".cc", ".cpp", ".hpp", ".rb", ".php", ".sh", ".bash",
              ".sql", ".html", ".css", ".scss", ".vue", ".svelte", ".swift",
              ".lua", ".r", ".jl", ".pl", ".cs", ".scala", ".dart", ".ex", ".exs")
_VERIFY_NUDGE = (
    "[auto-verify] You changed code but haven't run it. Verify the change actually works now: "
    "run the project's tests/build/linter if present, or otherwise execute the affected file or "
    "function (e.g. `pytest`, `npm test`, `python <file>`, `node <file>`, `go build ./...`). "
    "If it fails, read the error, fix the root cause, and re-run SYM_EMDASH repeat until it passes or you're "
    "genuinely blocked (then say what's blocking). Use run_bash; the user approves commands.")


def _is_code_file(path):
    return str(path or "").lower().endswith(_CODE_EXTS)


def _detect_project_commands(cwd=None):
    """Inspect the working dir for build/test markers and return the preferred
    verification commands (pytest / package.json scripts / Makefile target /
    go.mod / Cargo / etc.), so verification targets the real toolchain."""
    cwd = cwd or os.getcwd()
    def has(*names):
        return any(os.path.exists(os.path.join(cwd, n)) for n in names)
    cmds = []
    # Python
    if has("pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini") or \
            os.path.isdir(os.path.join(cwd, "tests")):
        cmds.append("pytest -q")
    if has("manage.py"):
        cmds.append("python manage.py check")
    # Node / JS
    if has("package.json"):
        try:
            scripts = (json.load(open(os.path.join(cwd, "package.json"),
                                       encoding="utf-8")) or {}).get("scripts", {}) or {}
        except Exception:  # noqa: BLE001
            scripts = {}
        if "test" in scripts:
            cmds.append("npm test")
        if "build" in scripts:
            cmds.append("npm run build")
        if "lint" in scripts:
            cmds.append("npm run lint")
        if not scripts:
            cmds.append("npm install && npm start")
    # Go / Rust
    if has("go.mod"):
        cmds += ["go build ./...", "go test ./..."]
    if has("Cargo.toml"):
        cmds += ["cargo build", "cargo test"]
    # Make — prefer a test/check/build target if one exists
    if has("Makefile", "makefile"):
        mk = ""
        for n in ("Makefile", "makefile"):
            p = os.path.join(cwd, n)
            if os.path.isfile(p):
                try:
                    mk = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    mk = ""
                break
        targets = set(re.findall(r"(?m)^([A-Za-z0-9_.-]+):", mk))
        cmds.append(next((f"make {t}" for t in ("test", "check", "build", "all")
                          if t in targets), "make"))
    # JVM / others
    if has("pom.xml"):
        cmds.append("mvn -q -DskipTests package")
    if has("build.gradle", "build.gradle.kts"):
        cmds.append("./gradlew build")
    if has("Gemfile"):
        cmds.append("bundle exec rake")
    if has("CMakeLists.txt"):
        cmds.append("cmake -S . -B build && cmake --build build")
    if has("composer.json"):
        cmds.append("composer test")
    if has("docker-compose.yml", "docker-compose.yaml", "compose.yaml"):
        cmds.append("docker compose config")  # validate (full `up` is heavy)
    seen, out = set(), []
    for c in cmds:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:4]


def _project_hint():
    cmds = _detect_project_commands()
    return (" This project's toolchain looks like: " + ", ".join(f"`{c}`" for c in cmds)
            + " SYM_EMDASH prefer those to verify.") if cmds else ""


def _git_context():
    """Return branch/status/recent commits for the cwd git repo, or empty string."""
    try:
        r = subprocess.run("git rev-parse --is-inside-work-tree",
                           shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if r.returncode != 0:
            return ""
        branch = subprocess.run("git branch --show-current", shell=True,
                                capture_output=True, text=True, cwd=os.getcwd()).stdout.strip()
        status = subprocess.run("git status --short", shell=True,
                                capture_output=True, text=True, cwd=os.getcwd()).stdout.strip()
        log = subprocess.run("git log --oneline -5", shell=True,
                             capture_output=True, text=True, cwd=os.getcwd()).stdout.strip()
        parts = [f"branch: {branch or '(detached HEAD)'}"]
        if status:
            parts.append(f"dirty files:\n{status}")
        if log:
            parts.append(f"recent commits:\n{log}")
        return "\n".join(parts)
    except Exception:
        return ""


def _verify_nudge():
    return _VERIFY_NUDGE + _project_hint()


# Phrases that mean "run / test / verify this (existing) project" — so Hera
# verifies a codebase it didn't write when you ask it to, like Claude Code.
_RUN_VERIFY_RE = re.compile(
    r"\b("
    r"run (it|this|the (project|code|app|tests?|server|system|build|suite|file|script)|my )|"
    r"make sure (it|this|the \w+) (runs?|works?|builds?|passes?|compiles?)|"
    r"(get|make) (it|this|the \w+) (run|running|work|working|build|building|pass|passing)|"
    r"does (it|this|the \w+) (run|build|work|compile|pass)|"
    r"verify (it|that|this|the )|"
    r"check (that |if )?(it|the \w+) (runs?|works?|builds?|compiles?)|"
    r"run the tests?|test (it|the (project|code|system|app|build))|"
    r"build (it|the (project|app|code))|is (it|the \w+) working|"
    r"see if (it|the \w+) (runs?|works?|builds?)"
    r")\b", re.IGNORECASE)


def _wants_run_verification(text):
    return bool(_RUN_VERIFY_RE.search(text or ""))


# Phrases that mean "change the code" — so when the model ends a turn having only
# *described* a fix (a weak/local model often narrates a patch instead of emitting
# an edit_file call), Hera nudges it once to actually apply the edit. Mirrors the
# verify-your-work loop, but for the edit step that precedes it.
_CHANGE_RE = re.compile(
    r"\b("
    r"fix|edit|change|modif(y|ies|ied)|update|patch|implement|refactor|rewrite|"
    r"add|create|append|insert|remove|delete|replace|rename|correct|resolve|"
    r"repair|adjust|tweak|convert|migrate|wire up|hook up|make (it|this|them)"
    r")\b", re.IGNORECASE)


def _wants_code_change(text):
    return bool(_CHANGE_RE.search(text or ""))


_EDIT_NUDGE = (
    "[auto-apply] Your turn ended without any file change, but the request asked for one. "
    "If you proposed or described a fix, you must actually apply it SYM_EMDASH call edit_file (exact "
    "old_string/new_string) or write_file now. Describing a patch in prose changes nothing on "
    "disk. If the task genuinely needs no edit, say so in one line and stop.")


# In-task to-do list (Claude-Code-style). Items: {"content": str, "status": ...}.
TODOS = []

# Background run_bash jobs: id -> {"proc", "out_path", "command", "started"}.
_BG_JOBS = {}
_BG_SEQ = 0

# Where custom slash-commands and named sub-agents live.
CONFIG_DIR   = os.path.dirname(CONFIG_PATH) or os.path.expanduser("~/.config/hera")
COMMANDS_DIR = os.path.join(CONFIG_DIR, "commands")
AGENTS_DIR   = os.path.join(CONFIG_DIR, "agents")

# Set (by the ESC watcher thread in the CLI, or a {"type":"interrupt"} message in
# --serve mode) to ask the current model turn to stop mid-stream. Cleared at the
# start of every turn. See interruptible() and stream_turn().
_INTERRUPT = threading.Event()
_VISION_WARNED = False  # warn once per process when an image can't be interpreted

# Thread-local "quiet" flag: parallel sub-agents run on worker threads and must
# not interleave their live streaming on the shared stdout. When set, stream_turn
# and _exec_call suppress live output (the sub-agent's result is still returned),
# and the approval gate stops prompting (treats writes like a non-interactive run).
_QUIET = threading.local()


def _is_quiet():
    return getattr(_QUIET, "on", False)


def _text_of(content):
    """Flatten a message's `content` to plain text.

    Content is normally a string, but a user message with an attached image is
    an OpenAI multimodal list of parts. Anything that assumes plain text (the
    compactor, session previews) goes through here.
    """
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                out.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                out.append("[image]")
        return " ".join(out).strip()
    return content or ""


def _msg_has_image(m):
    """True if a message's content carries an image_url part."""
    c = m.get("content")
    return isinstance(c, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in c)


def _downconvert_images(messages):
    """Replace image parts with a text placeholder so a text-only model can
    still answer the turn (it just can't see the picture)."""
    out = []
    for m in messages:
        if not _msg_has_image(m):
            out.append(m)
            continue
        texts = []
        for p in m["content"]:
            if p.get("type") == "text":
                texts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                texts.append("[image attached SYM_EMDASH not interpreted by this text-only model]")
        nm = dict(m)
        nm["content"] = "\n".join(t for t in texts if t).strip() or "[image attached]"
        out.append(nm)
    return out


def _select_endpoint(messages):
    """Pick (url, model, messages, downgraded) for a turn.

    If any message carries an image and HERA_VISION_URL is set, route to that
    vision endpoint/model. If an image is present but no vision endpoint is
    configured, down-convert the images to text and flag `downgraded` so the
    caller can warn that the current model can't see them.
    """
    has_image = any(_msg_has_image(m) for m in messages)
    if has_image and VISION_URL:
        return VISION_URL, (VISION_MODEL or MODEL), messages, False
    if has_image:
        return API_URL, MODEL, _downconvert_images(messages), True
    return API_URL, MODEL, messages, False


# ── Sandbox detection ─────────────────────────────────────────────────────────
def _detect_sandbox():
    """Pick the best available run_bash sandbox: bwrap > unshare > none."""
    if SANDBOX_MODE == "none":
        return "none"
    have_bwrap = bool(shutil.which("bwrap"))
    have_unshare = sys.platform.startswith("linux") and bool(shutil.which("unshare"))
    if SANDBOX_MODE == "bwrap":
        return "bwrap" if have_bwrap else "none"
    if SANDBOX_MODE == "unshare":
        return "unshare" if have_unshare else "none"
    # auto
    if have_bwrap:
        return "bwrap"
    if have_unshare:
        return "unshare"
    return "none"


SANDBOX_KIND = _detect_sandbox()

# Additional trusted directories beyond cwd (Claude-Code `--add-dir`): readable
# by the file tools and bind-mounted writable into the run_bash sandbox.
def _initial_extra_dirs():
    raw = _env("HERA_ADD_DIR", default="")
    dirs = [d for d in re.split(r"[:,]", raw) if d.strip()]
    cfg = _FILE_CFG.get("add_dirs")
    if isinstance(cfg, list):
        dirs += [str(d) for d in cfg]
    out = []
    for d in dirs:
        p = os.path.realpath(os.path.expanduser(d.strip()))
        if os.path.isdir(p) and p not in out:
            out.append(p)
    return out


EXTRA_DIRS = _initial_extra_dirs()


def _extra_binds():
    """bwrap --bind args for each extra trusted dir (writable)."""
    args = []
    for d in EXTRA_DIRS:
        if os.path.isdir(d):
            args += ["--bind", d, d]
    return args


def add_extra_dir(path):
    """Grant the session access to an extra directory (run_bash sandbox + tools)."""
    p = os.path.realpath(os.path.expanduser((path or "").strip()))
    if not os.path.isdir(p):
        return f"[error] not a directory: {p}"
    if p not in EXTRA_DIRS:
        EXTRA_DIRS.append(p)
    return f"added trusted dir: {p}  ({len(EXTRA_DIRS)} total)"


def sandbox_label():
    net = "network on" if SANDBOX_NET else "no network"
    extra = f" (+{len(EXTRA_DIRS)} added dir(s))" if EXTRA_DIRS else ""
    if SANDBOX_KIND == "bwrap":
        return f"bwrap {SYM_EMDASH} fs confined to cwd{extra}, {net}"
    if SANDBOX_KIND == "unshare":
        return f"unshare {SYM_EMDASH} pid-isolated, {net} (install bubblewrap for fs confinement)"
    return "none SYM_EMDASH run_bash runs unconfined"


def _sandbox_argv(command):
    """Return (argv, use_shell) to execute `command` under the active sandbox."""
    cwd = os.getcwd()
    # sudo requires setuid escalation; bwrap always sets no_new_privs which
    # permanently blocks setuid inside the sandbox. Run sudo commands unsandboxed
    # (they still go through the normal approval gate before reaching here).
    stripped = command.lstrip()
    if stripped.startswith("sudo ") or stripped == "sudo":
        return command, True
    if SANDBOX_KIND == "bwrap":
        # Order matters (later mounts win): make everything read-only, give a
        # private /tmp, then bind the working dir writable LAST so it stays
        # writable even when cwd is itself under /tmp.
        argv = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--bind", cwd, cwd, *_extra_binds(),
                "--die-with-parent", "--chdir", cwd]
        if not SANDBOX_NET:
            argv += ["--unshare-net"]
        argv += ["--", "/bin/sh", "-c", command]
        return argv, False
    if SANDBOX_KIND == "unshare":
        argv = ["unshare", "--user", "--map-root-user", "--fork", "--pid", "--mount-proc"]
        if not SANDBOX_NET:
            argv += ["--net"]
        argv += ["--", "/bin/sh", "-c", command]
        return argv, False
    return command, True  # none SYM_ARROW_R shell string


def _sandbox_wrap_argv(argv):
    """Wrap a program's argv under the active sandbox (for long-running children
    like MCP servers). Returns argv unchanged when sandboxing is off."""
    cwd = os.getcwd()
    if SANDBOX_KIND == "bwrap":
        pre = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--bind", cwd, cwd, *_extra_binds(),
               "--die-with-parent", "--chdir", cwd]
        if not SANDBOX_NET:
            pre += ["--unshare-net"]
        return pre + ["--"] + list(argv)
    if SANDBOX_KIND == "unshare":
        pre = ["unshare", "--user", "--map-root-user", "--fork", "--pid", "--mount-proc"]
        if not SANDBOX_NET:
            pre += ["--net"]
        return pre + ["--"] + list(argv)
    return list(argv)


# ── Permission allowlist ──────────────────────────────────────────────────────
# Patterns are fnmatch-style, matched against the full run_bash command string.
# A command auto-approves only if it matches ALLOW and not DENY. DENY always wins.
DENY_DEFAULTS = [
    "*rm -rf /*", "*rm -fr /*", "* rm -rf ~*", "*mkfs*", "*dd *of=/dev/*",
    "*:(){*", "*shutdown*", "*reboot*", "*>* /dev/sd*", "*chmod -R 777 /*",
    "*curl*|*sh*", "*wget*|*sh*",
]


def _split_patterns(raw):
    return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]


def _load_allow():
    pats = _split_patterns(_env("HERA_ALLOW"))
    f = os.path.join(os.getcwd(), ".heraallow")
    if os.path.isfile(f):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        pats.append(line)
        except OSError:
            pass
    return pats


ALLOW_PATTERNS = _load_allow()                       # auto-approve run_bash matches
DENY_PATTERNS  = DENY_DEFAULTS + _split_patterns(_env("HERA_DENY"))


def _matches(cmd, patterns):
    norm = " ".join(cmd.split())  # collapse whitespace
    return any(fnmatch.fnmatch(norm, p) for p in patterns)


def bash_allowed(cmd):
    """True if the command is pre-approved by the allowlist (and not denied)."""
    if _matches(cmd, DENY_PATTERNS):
        return False
    return _matches(cmd, ALLOW_PATTERNS)

# ── Theme ─────────────────────────────────────────────────────────────────────
# Colour is on for interactive terminals; disabled for pipes, NO_COLOR, or
# HERA_NO_COLOR. HERA_FORCE_COLOR=1 forces it on (useful for demos/screenshots).
USE_COLOR = (
    _truthy(_env("HERA_FORCE_COLOR"))
    or (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        and not _truthy(_env("HERA_NO_COLOR")))
)


def _sgr(code):
    return f"\033[{code}m" if USE_COLOR else ""


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s):
    return _ANSI_RE.sub("", s)


R     = _sgr("0")
BOLD  = _sgr("1")
DIM   = _sgr("2")
ITAL  = _sgr("3")
CYAN  = _sgr("96")
GREEN = _sgr("92")
RED   = _sgr("91")
YELL  = _sgr("93")
BLUE  = _sgr("94")
MAG   = _sgr("95")
GREY  = _sgr("90")
REV   = _sgr("7")
TEAL  = _sgr("38;5;44")
SKY   = _sgr("38;5;39")
IND    = _sgr("38;5;33")
ACCENT = CYAN

# ── UTF-8 capability detection + display symbol constants ─────────────────────
# Some terminals (older SSH sessions, minimal VMs) are not set to UTF-8. Every
# multi-byte Unicode glyph (block chars, box-drawing, braille spinner, bullets)
# renders as garbled bytes on those terminals. We detect this once at startup and
# fall back to plain ASCII equivalents everywhere in the UI.
import locale as _locale
def _detect_utf8():
    # HERA_ASCII=1 is a manual override for when the SSH client terminal is not
    # set to UTF-8 but the server LANG still says en_US.UTF-8 (server env vars
    # can't detect the client's terminal encoding). Set in ~/.bashrc on the remote
    # host: export HERA_ASCII=1
    if os.environ.get("HERA_ASCII", "").strip() in ("1", "true", "yes"):
        return False
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        val = os.environ.get(var, "").lower().replace("-", "")
        if val:
            return "utf8" in val or "utf" in val
    enc = (_locale.getpreferredencoding(False) or "").lower().replace("-", "")
    return "utf8" in enc or "utf" in enc
_UTF8 = _detect_utf8()

def _u(u, a):
    """Return the UTF-8 glyph if the terminal supports it, else the ASCII fallback."""
    return u if _UTF8 else a

# Box-drawing / block chars
SYM_HLINE    = _u("─", "-")
SYM_VLINE    = _u("│", "|")
SYM_TL       = _u("┌", "+")
SYM_BL       = _u("└", "+")
SYM_HALF_L   = _u("▌", "|")     # response / plan / doctor header accent
SYM_ACCENT   = _u("▎", "|")     # banner info-row accent
SYM_THIN     = _u("▏", "|")     # blockquote bar
SYM_PROG_F   = _u("█", "#")     # progress-bar full block
SYM_PROG_E   = _u("░", ".")     # progress-bar empty block
# Punctuation / directional
SYM_BULLET   = _u("•", "*")
SYM_MIDDOT   = _u("·", ".")
SYM_EMDASH   = _u("—", "--")
SYM_ENDASH   = _u("–", "-")
SYM_ELLIPSIS = _u("…", "...")
SYM_PROMPT   = _u("❯", ">")
SYM_ARROW_R  = _u("→", "->")
SYM_ARROW_U  = _u("↑", "^")
SYM_ARROW_D  = _u("↓", "v")
SYM_ARROW_L  = _u("←", "<-")
SYM_HOOKED   = _u("↳", ">")
SYM_SUB      = _u("⤷", ">")
SYM_RECYCLE  = _u("⟳", "~")
SYM_DIAMOND  = _u("◆", "+")
SYM_SUBARROW = _u("›", ">")
# Status marks
SYM_CHECK    = _u("✓", "+")
SYM_DONE     = _u("✔", "+")
SYM_CROSS    = _u("✗", "x")
SYM_WARN     = _u("⚠", "!")
SYM_THINK    = _u("✶", "*")
SYM_INTERRUPT= _u("⎿", "\\")
# Todo
SYM_PENDING  = _u("○", "o")
SYM_INPROG   = _u("▸", ">")
# Wordmark — pixel-art HERA; █ → # in ASCII mode
_WORDMARK = [
    _u("█  █  ████  ███    ██ ",  "#  #  ####  ###    ## "),
    _u("█  █  █     █  █  █  █ ", "#  #  #     #  #  #  # "),
    _u("████  ███   ███   ████ ",  "####  ###   ###   #### "),
    _u("█  █  █     █ █   █  █ ", "#  #  #     # #   #  # "),
    _u("█  █  ████  █  █  █  █ ", "#  #  ####  #  #  #  # "),
]
# Spinner frames — braille in UTF-8, classic pipe in ASCII mode
_SPINNER_FRAMES = _u("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", r"|/-\|/-\|/")


# ── Lightweight markdown rendering (per completed line) ────────────────────────
def _md_inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{R}", s)
    s = re.sub(r"`([^`]+)`", lambda m: f"{CYAN}{m.group(1)}{R}", s)
    return s


def render_md_line(line, state):
    """Style one completed markdown line. `state` tracks fenced-code blocks."""
    try:
        stripped = line.rstrip()
        if stripped.lstrip().startswith("```"):
            state["code"] = not state["code"]
            lang = stripped.lstrip()[3:].strip()
            if state["code"]:
                return f"  {GREY}┌─ {lang or 'code'} {SYM_HLINE * max(0, 40 - len(lang))}{R}"
            return f"  {GREY}└{SYM_HLINE * 46}{R}"
        if state["code"]:
            return f"  {GREY}{SYM_VLINE}{R} {line}"
        if re.match(r"^#{1,6}\s", stripped):
            _heading_text = re.sub(r"^#{1,6}\s*", "", stripped)
            return f"{BOLD}{CYAN}{_md_inline(_heading_text)}{R}"
        m = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if m:
            return f"{m.group(1)}  {CYAN}{SYM_BULLET}{R} {_md_inline(m.group(2))}"
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)", line)
        if m:
            return f"{m.group(1)}  {CYAN}{m.group(2)}.{R} {_md_inline(m.group(3))}"
        if stripped.startswith(">"):
            return f"  {GREY}{SYM_THIN}{R} {DIM}{stripped[1:].strip()}{R}"
        return _md_inline(line)
    except Exception:  # noqa: BLE001 SYM_EMDASH never let rendering break the stream
        return line


# ── Spinner ───────────────────────────────────────────────────────────────────
# Whimsical present-progressive status words shown while Hera is busy — the same
# trick Claude Code uses so a long operation feels alive rather than frozen. One
# is picked at random and swapped for another every few seconds.
_WHIMSY = [
    "Thinking", "Pondering", "Noodling", "Cogitating", "Ruminating",
    "Percolating", "Conjuring", "Tinkering", "Scheming", "Marinating",
    "Synthesizing", "Untangling", "Wrangling", "Spelunking", "Calibrating",
    "Finagling", "Whirring", "Computing", "Brewing", "Deliberating",
    "Puzzling", "Contemplating", "Crunching", "Assembling", "Orchestrating",
    "Mulling", "Hatching", "Tessellating", "Sleuthing", "Reticulating",
]


class Spinner:
    _FRAMES = _SPINNER_FRAMES
    _ROTATE_EVERY = 4.0    # seconds between whimsy-word changes

    def __init__(self):
        self._stop = threading.Event()
        self._t    = None
        self._t0   = None

    def _run(self, label):
        # label=None → rotate random whimsy words; an explicit label stays put.
        whimsy = label is None
        word = random.choice(_WHIMSY) if whimsy else label
        last_rot = time.time()
        i = 0
        while not self._stop.is_set():
            now = time.time()
            secs = now - self._t0
            if whimsy and now - last_rot >= self._ROTATE_EVERY:
                word = random.choice(_WHIMSY)
                last_rot = now
            f = self._FRAMES[i % len(self._FRAMES)]
            tail = "SYM_ELLIPSIS" if whimsy else ""
            print(f"\r  {CYAN}{f}{R}  {DIM}{word}{tail} {secs:.1f}s{R}   ", end="", flush=True)
            i += 1
            time.sleep(0.1)

    def start(self, label=None):
        self._t0 = time.time()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, args=(label,), daemon=True)
        self._t.start()

    def stop(self):
        elapsed = time.time() - self._t0 if self._t0 else 0
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join()
        print(f"\r{' ' * 48}\r", end="", flush=True)
        return elapsed


# ── Tools ─────────────────────────────────────────────────────────────────────
def _resolve(path):
    return os.path.abspath(os.path.expanduser(path))


def tool_list_dir(path="."):
    p = _resolve(path)
    if not os.path.isdir(p):
        return f"[error] not a directory: {p}"
    entries = sorted(os.listdir(p))
    if not entries:
        return f"(empty directory: {p})"
    lines = []
    for name in entries:
        full = os.path.join(p, name)
        suffix = "/" if os.path.isdir(full) else ""
        lines.append(f"{name}{suffix}")
    return f"{p}:\n" + "\n".join(lines)


def tool_read_file(path, offset=None, limit=None):
    p = _resolve(path)
    if not os.path.isfile(p):
        return f"[error] no such file: {p}"
    low = p.lower()
    if low.endswith(".ipynb"):
        return _render_notebook(p)
    if low.endswith(".pdf"):
        text = _read_pdf(p)
        if text is None:
            return ("[error] couldn't extract PDF text SYM_EMDASH install poppler "
                    "(`pdftotext`) or the `pypdf` package, then retry.")
        return text[:MAX_READ_BYTES]
    if os.path.getsize(p) > MAX_READ_BYTES:
        return f"[error] file too large (> {MAX_READ_BYTES} bytes); read a slice with run_bash sed/head"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = (int(offset) - 1) if offset else 0
    start = max(start, 0)
    end = (start + int(limit)) if limit else len(lines)
    out = []
    for i in range(start, min(end, len(lines))):
        out.append(f"{i + 1:>6}\t{lines[i].rstrip(chr(10))}")
    if not out:
        return "(no lines in range)"
    return "\n".join(out)


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             ".mypy_cache", ".pytest_cache", "dist", "build"}


def tool_glob(pattern, path="."):
    base = _resolve(path)
    if not os.path.isdir(base):
        return f"[error] not a directory: {base}"
    matches = [m for m in globmod.glob(os.path.join(base, pattern), recursive=True)
               if os.path.isfile(m)]
    matches.sort(key=os.path.getmtime, reverse=True)  # most recent first
    if not matches:
        return f"(no files match {pattern!r} under {base})"
    head = matches[:200]
    out = "\n".join(head)
    if len(matches) > 200:
        out += f"\n{SYM_ELLIPSIS}[{len(matches) - 200} more]"
    return out


def tool_search(pattern, path=".", glob="*", ignore_case=False, max_results=200):
    base = _resolve(path)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"[error] bad regex: {exc}"
    results = []
    targets = []
    if os.path.isfile(base):
        targets = [base]
    else:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                if fnmatch.fnmatch(fn, glob):
                    targets.append(os.path.join(root, fn))
    for fp in targets:
        try:
            if os.path.getsize(fp) > MAX_READ_BYTES:
                continue
            with open(fp, "r", encoding="utf-8") as f:  # strict: skips binaries
                for i, line in enumerate(f, 1):
                    if rx.search(line):
                        results.append(f"{fp}:{i}:{line.rstrip(chr(10))[:200]}")
                        if len(results) >= int(max_results):
                            results.append(f"{SYM_ELLIPSIS}[stopped at {max_results} matches]")
                            return "\n".join(results)
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable SYM_EMDASH skip
    if not results:
        return f"(no matches for /{pattern}/ under {base})"
    return "\n".join(results)


CODE_EXTS = ("py", "js", "ts", "jsx", "tsx", "go", "rs", "java", "c", "cc",
             "cpp", "h", "hpp", "rb", "sh", "php", "cs", "kt", "swift")
_DEF_RE = re.compile(
    r"\b(?:func|function|class|def|fn|type|interface|struct|impl|module)\s+([A-Za-z_]\w*)")


def _code_files(base):
    if os.path.isfile(base):
        return [base]
    out = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.rsplit(".", 1)[-1] in CODE_EXTS:
                out.append(os.path.join(root, fn))
    return out


def tool_symbols(name=None, path="."):
    """Codebase index: list function/class/etc. definitions (optionally filtered)."""
    base = _resolve(path)
    if not os.path.exists(base):
        return f"[error] no such path: {base}"
    results = []
    for fp in _code_files(base):
        try:
            if os.path.getsize(fp) > MAX_READ_BYTES:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if fp.endswith(".py"):
            try:
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        kind = "class" if isinstance(node, ast.ClassDef) else "def"
                        results.append((fp, node.lineno, f"{kind} {node.name}"))
                continue
            except SyntaxError:
                pass  # fall through to regex
        for i, line in enumerate(text.splitlines(), 1):
            m = _DEF_RE.search(line)
            if m:
                results.append((fp, i, line.strip()[:100]))
    if name:
        nl = name.lower()
        results = [r for r in results if nl in r[2].lower()]
    if not results:
        return f"(no symbols{' matching ' + repr(name) if name else ''} under {base})"
    results.sort()
    out = "\n".join(f"{fp}:{ln}: {sig}" for fp, ln, sig in results[:300])
    if len(results) > 300:
        out += f"\n{SYM_ELLIPSIS}[{len(results) - 300} more]"
    return out


# ── Semantic search (embeddings-backed; registered only if available) ──────────
EMBED_URL   = _env("HERA_EMBED_URL", default=API_URL)
EMBED_MODEL = _env("HERA_EMBED_MODEL", default=MODEL)


def _embed(texts):
    """Return a list of embedding vectors for `texts` via an OpenAI-compatible API."""
    resp = requests.post(
        f"{EMBED_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def embeddings_available():
    try:
        v = _embed(["ping"])
        return bool(v and v[0])
    except Exception:  # noqa: BLE001
        return False


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def tool_semantic_search(query, path=".", k=8):
    """Rank code chunks by embedding similarity to the query."""
    base = _resolve(path)
    chunks = []  # (file, start_line, text)
    for fp in _code_files(base):
        try:
            if os.path.getsize(fp) > MAX_READ_BYTES:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i in range(0, len(lines), 20):
            block = "\n".join(lines[i:i + 20]).strip()
            if block:
                chunks.append((fp, i + 1, block[:1200]))  # keep within embed ctx
        if len(chunks) >= 600:
            break
    if not chunks:
        return f"(no code to search under {base})"
    try:
        qvec = _embed([query[:1200]])[0]
        vecs = []
        texts = [c[2] for c in chunks]
        for j in range(0, len(texts), 32):  # small batches SYM_ARROW_R stay under embed ctx
            vecs.extend(_embed(texts[j:j + 32]))
    except Exception as exc:  # noqa: BLE001
        return f"[error] embeddings request failed: {exc}"
    scored = sorted(zip(chunks, vecs), key=lambda cv: _cosine(qvec, cv[1]), reverse=True)
    lines_out = []
    for (fp, ln, text), vec in scored[:int(k)]:
        snippet = text.splitlines()[0][:100] if text else ""
        lines_out.append(f"{fp}:{ln}: ({_cosine(qvec, vec):.2f}) {snippet}")
    return "\n".join(lines_out)


def tool_write_file(path, content):
    p = _resolve(path)
    parent = os.path.dirname(p)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    existed = os.path.isfile(p)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {p} ({n} lines)"


def tool_edit_file(path, old_string, new_string, replace_all=False):
    p = _resolve(path)
    if not os.path.isfile(p):
        return f"[error] no such file: {p}"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    count = text.count(old_string)
    if count == 0:
        return "[error] old_string not found in file (must match exactly)"
    if count > 1 and not replace_all:
        replace_all = True  # identical matches: promote silently
    new_text = text.replace(old_string, new_string)
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_text)
    return f"Edited {p} ({count} replacement{'s' if count != 1 else ''})"


def tool_multi_edit(path, edits):
    """Apply several old/new replacements to one file atomically (all-or-nothing).

    `edits` is a list of {old_string, new_string, replace_all?}. Each is applied
    in order to the running text; if any fails (not found / not unique) nothing is
    written and the failing edit is reported. Mirrors Claude Code's MultiEdit."""
    p = _resolve(path)
    if not os.path.isfile(p):
        return f"[error] no such file: {p}"
    if not isinstance(edits, list) or not edits:
        return "[error] edits must be a non-empty list of {old_string, new_string}"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    applied = 0
    for i, e in enumerate(edits):
        if not isinstance(e, dict):
            return f"[error] edit #{i + 1} is not an object"
        old_s = e.get("old_string", "")
        new_s = e.get("new_string", "")
        replace_all = bool(e.get("replace_all", False))
        if old_s == "":
            return f"[error] edit #{i + 1}: old_string is empty"
        count = text.count(old_s)
        if count == 0:
            return f"[error] edit #{i + 1}: old_string not found (no change written)"
        if count > 1 and not replace_all:
            replace_all = True  # identical matches: promote silently
        text = text.replace(old_s, new_s) if replace_all else text.replace(old_s, new_s, 1)
        applied += count if replace_all else 1
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Edited {p} ({len(edits)} edits, {applied} replacement{'s' if applied != 1 else ''})"


def _read_pdf(p):
    """Extract text from a PDF: prefer `pdftotext` (poppler), fall back to a
    pure-Python pypdf/PyPDF2 if importable. Returns text or None."""
    if shutil.which("pdftotext"):
        proc = subprocess.run(["pdftotext", "-layout", p, "-"],
                              capture_output=True, text=True)
        if proc.returncode == 0 and (proc.stdout or "").strip():
            return proc.stdout
    for mod in ("pypdf", "PyPDF2"):
        try:
            m = __import__(mod)
        except ImportError:
            continue
        try:
            reader = m.PdfReader(p)
            return "\n".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception:  # noqa: BLE001 SYM_EMDASH try the next backend
            continue
    return None


def _render_notebook(p):
    """Render a .ipynb as readable text: each cell's index, type, source, and a
    truncated view of its outputs."""
    try:
        with open(p, encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return f"[error] cannot read notebook: {exc}"
    out = []
    for i, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type", "?")
        src = "".join(cell.get("source", []))
        out.append(f"=== cell [{i}] ({ctype}) ===")
        out.append(src or "(empty)")
        for o in (cell.get("outputs") or []):
            ot = o.get("output_type")
            if ot == "stream":
                out.append("  --- stream ---\n" + "".join(o.get("text", []))[:1000])
            elif ot in ("execute_result", "display_data"):
                txt = "".join((o.get("data") or {}).get("text/plain", []))
                if txt:
                    out.append("  --- output ---\n" + txt[:1000])
            elif ot == "error":
                out.append("  --- error ---\n" + "\n".join(o.get("traceback", []))[:1000])
    return "\n".join(out) if out else "(empty notebook)"


def tool_notebook_edit(path, cell_index=None, new_source="", cell_type=None,
                       edit_mode="replace"):
    """Edit a Jupyter notebook cell. edit_mode: replace | insert | delete.
    Mirrors Claude Code's NotebookEdit."""
    p = _resolve(path)
    if not os.path.isfile(p):
        return f"[error] no such file: {p}"
    try:
        with open(p, encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return f"[error] cannot parse notebook: {exc}"
    cells = nb.setdefault("cells", [])
    n = len(cells)
    src_lines = new_source.splitlines(keepends=True) or ([new_source] if new_source else [])

    def _mk(ctype):
        c = {"cell_type": ctype, "metadata": {}, "source": src_lines}
        if ctype == "code":
            c["outputs"] = []
            c["execution_count"] = None
        return c

    if edit_mode == "delete":
        if not isinstance(cell_index, int) or not (0 <= cell_index < n):
            return f"[error] delete needs a valid cell_index 0..{n - 1}"
        cells.pop(cell_index)
        msg = f"deleted cell [{cell_index}]"
    elif edit_mode == "insert":
        idx = n if not isinstance(cell_index, int) else max(0, min(cell_index, n))
        cells.insert(idx, _mk(cell_type or "code"))
        msg = f"inserted {cell_type or 'code'} cell at [{idx}]"
    else:  # replace
        if not isinstance(cell_index, int) or not (0 <= cell_index < n):
            return f"[error] replace needs a valid cell_index 0..{n - 1}"
        cell = cells[cell_index]
        cell["source"] = src_lines
        if cell_type:
            cell["cell_type"] = cell_type
        if cell.get("cell_type") == "code":
            cell["outputs"] = []  # stale outputs no longer match the new source
        msg = f"replaced cell [{cell_index}]"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return f"Edited {p} ({msg})"


def _missing_program(command, stderr, returncode):
    """Detect a 'command not found' failure and return the missing binary name."""
    if returncode != 127:
        return None
    m = (re.search(r"([\w.+-]+): (?:command )?not found", stderr or "")
         or re.search(r"(?:command not found|not found): ([\w.+-]+)", stderr or ""))
    if m:
        return m.group(1)
    toks = command.split()
    return toks[0] if toks else None


def _install_plan(program):
    """Pick an install command for `program`, or None if no package manager fits."""
    pkg = _PKG_MAP.get(program, program)
    sudo = "" if getattr(os, "geteuid", lambda: 0)() == 0 else "sudo -n "
    if shutil.which("apt-get"):
        return f"{sudo}apt-get install -y {pkg}"
    if shutil.which("dnf"):
        return f"{sudo}dnf install -y {pkg}"
    if shutil.which("pacman"):
        return f"{sudo}pacman -S --noconfirm {pkg}"
    if shutil.which("brew"):
        return f"brew install {pkg}"
    if shutil.which("apk"):
        return f"{sudo}apk add {pkg}"
    return None


def _do_install(program):
    """Run the install plan for `program` (unsandboxed). Returns (ok, message).

    The package manager needs the network and writes to system dirs, so this is
    deliberately NOT run under the run_bash sandbox.
    """
    plan = _install_plan(program)
    if not plan:
        return False, f"no supported package manager found to install '{program}'"
    proc = subprocess.run(plan, shell=True, capture_output=True, text=True)
    if proc.returncode == 0 and shutil.which(program):
        return True, f"installed {program} {SYM_ARROW_R} {shutil.which(program)}"
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    hint = detail[-1] if detail else f"exit {proc.returncode}"
    if "password" in (proc.stderr or "").lower():
        hint = "needs sudo (no password available here) SYM_EMDASH install it manually"
    return False, f"install failed: {hint}"


# Headless (--serve) mode sets this so the reactive "missing binary" install
# offer surfaces as an IDE approval button instead of a terminal input() prompt.
# Signature: (program, plan) -> bool.
_INSTALL_APPROVER = None


def _confirm_install(program, plan):
    """Get consent to install `program`. YOLO auto-approves; in --serve mode it
    routes through the IDE approval hook; otherwise it prompts on the terminal."""
    if YOLO:
        return True
    if _INSTALL_APPROVER is not None:
        return bool(_INSTALL_APPROVER(program, plan))
    print(f"\n{YELL}{BOLD}{SYM_WARN} '{program}' is not installed.{R}", file=sys.stderr)
    print(f"  {DIM}proposed:{R} {plan}", file=sys.stderr)
    try:
        ans = input(f"{BOLD}  install it now? [y]es / [n]o:{R} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes", "")


def _offer_install(program):
    """Reactive path: a run_bash command hit a missing binary. Ask, then install."""
    if not AUTO_INSTALL:
        return False
    plan = _install_plan(program)
    if not plan:
        return False
    if not _confirm_install(program, plan):
        return False
    print(f"  {DIM}installing{SYM_ELLIPSIS} (outside the sandbox, with network){R}", file=sys.stderr)
    ok, msg = _do_install(program)
    print(f"  {GREEN}{SYM_CHECK} {msg}{R}" if ok else f"  {RED}{SYM_CROSS} {msg}{R}", file=sys.stderr)
    return ok


def tool_install(program, reason=""):
    """Proactive path: the model decided it needs `program`. Approval is handled
    by the side-effect gate before this runs, so here we just install it."""
    program = (program or "").strip().split()[0] if program else ""
    if not program:
        return "[error] no program specified"
    existing = shutil.which(program)
    if existing:
        return f"{program} is already installed ({existing})"
    # Progress goes to stderr so it never pollutes the --serve JSON stdout stream;
    # stderr is still shown in the terminal REPL.
    print(f"  {DIM}installing {program}{SYM_ELLIPSIS} (outside the sandbox, with network){R}",
          file=sys.stderr)
    ok, msg = _do_install(program)
    return msg if ok else f"[error] {msg}"


def tool_run_bash(command, timeout=3600, run_in_background=False, _retry=False):
    if run_in_background:
        return _bash_background_start(command)
    argv, use_shell = _sandbox_argv(command)
    try:
        proc = subprocess.run(
            argv, shell=use_shell, capture_output=True, text=True,
            timeout=int(timeout), cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except FileNotFoundError as exc:
        return f"[error] sandbox launch failed ({exc}); set HERA_SANDBOX=none to disable"
    out = proc.stdout or ""
    err = proc.stderr or ""

    # A missing binary → offer to install it and retry once, instead of just
    # handing the model a bare exit-127.
    if not _retry:
        prog = _missing_program(command, err, proc.returncode)
        if prog and _offer_install(prog):
            print(f"  {DIM}{SYM_HOOKED} re-running: {command}{R}", file=sys.stderr)
            return tool_run_bash(command, timeout, _retry=True)

    parts = []
    if out:
        parts.append(out.rstrip("\n"))
    if err:
        parts.append(f"[stderr]\n{err.rstrip(chr(10))}")
    parts.append(f"[exit code {proc.returncode}]")
    return "\n".join(parts)


# ── Web access (search + fetch) ───────────────────────────────────────────────
def _html_text(s):
    """Strip tags/entities from an HTML fragment and collapse whitespace."""
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", ihtml.unescape(s)).strip()


def _ddg_url(href):
    """DuckDuckGo wraps result links in a redirect SYM_EMDASH unwrap to the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = _urlparse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        q = _urlparse.parse_qs(parsed.query).get("uddg")
        if q:
            return _urlparse.unquote(q[0])
    return href


def _format_results(items, max_results):
    """Format a list of {title,url,snippet} dicts into the ranked text block."""
    out = []
    for i, it in enumerate(items[:max_results]):
        out.append(f"{i + 1}. {(it.get('title') or '').strip()}\n"
                   f"   {it.get('url', '')}\n   {(it.get('snippet') or '').strip()}")
    return "\n".join(out) if out else "(no results found)"


def _search_duckduckgo(query, n):
    resp = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                         headers={"User-Agent": _WEB_UA}, timeout=WEB_TIMEOUT)
    resp.raise_for_status()
    body = resp.text
    titles = re.findall(r'class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>', body, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.S)
    items = []
    for i, (href, title) in enumerate(titles[:n]):
        items.append({"title": _html_text(title), "url": _ddg_url(href),
                      "snippet": _html_text(snippets[i]) if i < len(snippets) else ""})
    return items


def _search_tavily(query, n):
    r = requests.post("https://api.tavily.com/search",
                      json={"api_key": SEARCH_KEY, "query": query, "max_results": n},
                      timeout=WEB_TIMEOUT)
    r.raise_for_status()
    return [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content")}
            for x in (r.json().get("results") or [])]


def _search_brave(query, n):
    r = requests.get("https://api.search.brave.com/res/v1/web/search",
                     params={"q": query, "count": n},
                     headers={"X-Subscription-Token": SEARCH_KEY,
                              "Accept": "application/json"}, timeout=WEB_TIMEOUT)
    r.raise_for_status()
    web = (r.json().get("web") or {}).get("results") or []
    return [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("description")}
            for x in web]


def _search_searxng(query, n):
    base = SEARCH_URL or "http://localhost:8888"
    r = requests.get(f"{base}/search", params={"q": query, "format": "json"},
                     headers={"User-Agent": _WEB_UA}, timeout=WEB_TIMEOUT)
    r.raise_for_status()
    return [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content")}
            for x in (r.json().get("results") or [])[:n]]


_SEARCH_PROVIDERS = {"duckduckgo": _search_duckduckgo, "tavily": _search_tavily,
                     "brave": _search_brave, "searxng": _search_searxng}


def tool_web_search(query, max_results=6):
    """Search the web and return ranked title/url/snippet results. Uses the
    configured provider (default DuckDuckGo); falls back to DuckDuckGo on error."""
    provider = _SEARCH_PROVIDERS.get(SEARCH_PROVIDER, _search_duckduckgo)
    try:
        items = provider(query, max_results)
    except requests.exceptions.RequestException as exc:
        if provider is not _search_duckduckgo:
            try:
                items = _search_duckduckgo(query, max_results)
            except requests.exceptions.RequestException as exc2:
                return f"[error] web search failed: {exc2}"
        else:
            return f"[error] web search failed: {exc}"
    return _format_results(items, max_results)


def tool_web_fetch(url, max_chars=9000):
    """Fetch a web page and return its readable text content."""
    if not re.match(r"https?://", url):
        url = "https://" + url
    try:
        resp = requests.get(url, headers={"User-Agent": _WEB_UA},
                            timeout=WEB_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return f"[error] fetch failed: {exc}"
    text = _html_text(resp.text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n{SYM_ELLIPSIS}[truncated, {len(text)} chars total]"
    return f"{url}\n\n{text}"


TOOLS = {
    "list_dir":   tool_list_dir,
    "read_file":  tool_read_file,
    "glob":       tool_glob,
    "search":     tool_search,
    "symbols":    tool_symbols,
    "write_file": tool_write_file,
    "edit_file":  tool_edit_file,
    "multi_edit": tool_multi_edit,
    "notebook_edit": tool_notebook_edit,
    "run_bash":   tool_run_bash,
}

# Tools that change the world → require approval (unless YOLO).
SIDE_EFFECTS = {"write_file", "edit_file", "multi_edit", "notebook_edit", "run_bash"}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries in a directory. Directories are suffixed with '/'.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path (default '.')"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file. Returns lines prefixed with line numbers (like cat -n).",
        "parameters": {"type": "object", "properties": {
            "path":   {"type": "string", "description": "File path"},
            "offset": {"type": "integer", "description": "1-based start line (optional)"},
            "limit":  {"type": "integer", "description": "Max lines to read (optional)"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "glob",
        "description": ("Find files whose path matches a glob pattern (use ** for recursion, "
                        "e.g. '**/*.py'). Returns paths, most recently modified first."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
            "path":    {"type": "string", "description": "Base directory to search from (default '.')"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "search",
        "description": ("Search file contents for a regular expression (like grep -rn). "
                        "Returns matching lines as path:line:text."),
        "parameters": {"type": "object", "properties": {
            "pattern":     {"type": "string", "description": "Python regular expression"},
            "path":        {"type": "string", "description": "Directory or file to search (default '.')"},
            "glob":        {"type": "string", "description": "Only search files matching this name glob, e.g. '*.py' (default '*')"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default false)"},
            "max_results": {"type": "integer", "description": "Cap on matches returned (default 200)"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "symbols",
        "description": ("Codebase index: list function/class/type definitions across the "
                        "project as path:line: signature. Pass `name` to filter. Great for "
                        "'where is X defined?'."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Only show symbols whose signature contains this (optional)"},
            "path": {"type": "string", "description": "Directory or file to index (default '.')"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content. Parent dirs are created.",
        "parameters": {"type": "object", "properties": {
            "path":    {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Full file content"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": ("Replace an exact substring in a file. old_string must match "
                        "exactly and be unique unless replace_all is true."),
        "parameters": {"type": "object", "properties": {
            "path":        {"type": "string"},
            "old_string":  {"type": "string", "description": "Exact text to replace"},
            "new_string":  {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
        }, "required": ["path", "old_string", "new_string"]},
    }},
    {"type": "function", "function": {
        "name": "multi_edit",
        "description": ("Apply several exact-string replacements to ONE file atomically "
                        "(all-or-nothing). Use this instead of multiple edit_file calls when "
                        "changing several places in the same file."),
        "parameters": {"type": "object", "properties": {
            "path":  {"type": "string"},
            "edits": {"type": "array", "description": "Edits applied in order",
                      "items": {"type": "object", "properties": {
                          "old_string":  {"type": "string"},
                          "new_string":  {"type": "string"},
                          "replace_all": {"type": "boolean"},
                      }, "required": ["old_string", "new_string"]}},
        }, "required": ["path", "edits"]},
    }},
    {"type": "function", "function": {
        "name": "notebook_edit",
        "description": ("Edit a Jupyter .ipynb cell. edit_mode 'replace' overwrites the cell at "
                        "cell_index, 'insert' adds a new cell at cell_index, 'delete' removes it. "
                        "Read the notebook first with read_file to see cell indices."),
        "parameters": {"type": "object", "properties": {
            "path":       {"type": "string"},
            "cell_index": {"type": "integer", "description": "0-based cell index"},
            "new_source": {"type": "string", "description": "New cell source (replace/insert)"},
            "cell_type":  {"type": "string", "enum": ["code", "markdown"],
                           "description": "Cell type for insert/replace (default code)"},
            "edit_mode":  {"type": "string", "enum": ["replace", "insert", "delete"],
                           "description": "Default replace"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": ("Run a shell command in the current working directory and return "
                        "stdout, stderr, and exit code."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 3600)"},
            "run_in_background": {"type": "boolean", "description": ("Start a long-running command "
                "(server, watcher, build) detached and return a job id immediately; read its output "
                "later with bash_output. Default false.")},
        }, "required": ["command"]},
    }},
]


# ── Approval gate ─────────────────────────────────────────────────────────────
_always_ok = set()  # tool names approved for the rest of the session


def _diff_preview(old, new):
    """A small unified-ish diff of an edit (capped)."""
    lines = []
    olines, nlines = old.splitlines(), new.splitlines()
    for ln in olines[:8]:
        lines.append(f"    {RED}- {ln[:80]}{R}")
    if len(olines) > 8:
        lines.append(f"    {RED}- {SYM_ELLIPSIS}{R}")
    for ln in nlines[:8]:
        lines.append(f"    {GREEN}+ {ln[:80]}{R}")
    if len(nlines) > 8:
        lines.append(f"    {GREEN}+ {SYM_ELLIPSIS}{R}")
    return "\n".join(lines)


def _full_diff(old, new, max_lines=120):
    """A colored unified diff of the *whole* proposed change (capped).

    Shown before an edit is applied so the user sees exactly what will change SYM_EMDASH
    Claude-Code-style. `old`/`new` are the full before/after file contents.
    """
    a = old.splitlines()
    b = new.splitlines()
    out = []
    for ln in difflib.unified_diff(a, b, lineterm="", n=2):
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("@@"):
            out.append(f"    {CYAN}{ln}{R}")
        elif ln.startswith("+"):
            out.append(f"    {GREEN}{ln[:200]}{R}")
        elif ln.startswith("-"):
            out.append(f"    {RED}{ln[:200]}{R}")
        else:
            out.append(f"    {DIM}{ln[:200]}{R}")
    if not out:
        return f"    {DIM}(no textual change){R}"
    if len(out) > max_lines:
        extra = len(out) - max_lines
        out = out[:max_lines] + [f"    {DIM}{SYM_ELLIPSIS} (+{extra} more diff lines){R}"]
    return "\n".join(out)


def _narrate(name, args):
    """A plain-language, one-line description of what a tool call is about to do.

    Printed before each tool runs so a reader can follow Hera's actions in
    words, not just see raw tool cards.
    """
    p = args.get("path", "")
    if name == "read_file":
        return f"Reading {p}"
    if name == "write_file":
        return f"Writing {p}"
    if name == "edit_file":
        return f"Editing {p}"
    if name == "multi_edit":
        return f"Editing {p} ({len(args.get('edits') or [])} edits)"
    if name == "notebook_edit":
        return f"{args.get('edit_mode', 'replace').title()} notebook cell in {p}"
    if name == "list_dir":
        return f"Listing {args.get('path', '.')}"
    if name == "glob":
        return f"Finding files matching {args.get('pattern', '')}"
    if name == "search":
        return f"Searching code for {args.get('pattern', '')}"
    if name == "symbols":
        return "Indexing code symbols"
    if name == "semantic_search":
        return f"Semantic-searching for {args.get('query', '')}"
    if name == "run_bash":
        return f"Running  {args.get('command', '')}"
    if name == "install_tool":
        return f"Installing {args.get('program', '?')}"
    if name == "web_search":
        return f"Searching the web for {args.get('query', '')}"
    if name == "web_fetch":
        return f"Fetching {args.get('url', '')}"
    if name == "task":
        return f"Delegating a sub-task: {args.get('description', '')[:60]}"
    return f"Calling {name}"


def _preview_call(name, args):
    if name == "run_bash":
        return f"$ {args.get('command', '')}"
    if name == "install_tool":
        prog = args.get("program", "?")
        reason = args.get("reason", "")
        plan = _install_plan(prog) or "no package manager available"
        head = f"install {prog}" + (f"  {SYM_EMDASH} {reason}" if reason else "")
        return f"{head}\n    $ {plan}"
    if name == "write_file":
        path = args.get("path", "?")
        c = args.get("content", "")
        n = c.count("\n") + 1
        existed, before = _snapshot(path)
        verb = "overwrite" if existed else "create"
        return (f"{verb} {path}  ({n} lines)\n"
                f"{_full_diff(before or '', c)}")
    if name == "edit_file":
        path = args.get("path", "?")
        old_s, new_s = args.get("old_string", ""), args.get("new_string", "")
        existed, before = _snapshot(path)
        # Compute the proposed after-text without touching the file, so we can
        # show the full change before it is applied.
        if existed and before is not None and old_s in before:
            replace_all = args.get("replace_all", False)
            after = (before.replace(old_s, new_s) if replace_all
                     else before.replace(old_s, new_s, 1))
            return f"edit {path}\n{_full_diff(before, after)}"
        return f"edit {path}\n{_diff_preview(old_s, new_s)}"
    if name == "multi_edit":
        path = args.get("path", "?")
        edits = args.get("edits") or []
        existed, before = _snapshot(path)
        # Apply the edits to a copy so we can show one combined diff up front.
        if existed and before is not None:
            after, ok = before, True
            for e in edits:
                old_s = e.get("old_string", "")
                if old_s and old_s in after:
                    after = (after.replace(old_s, e.get("new_string", ""))
                             if e.get("replace_all") else after.replace(old_s, e.get("new_string", ""), 1))
                else:
                    ok = False
                    break
            if ok:
                return f"multi-edit {path} ({len(edits)} edits)\n{_full_diff(before, after)}"
        return f"multi-edit {path} ({len(edits)} edits)"
    if name == "notebook_edit":
        mode = args.get("edit_mode", "replace")
        return (f"{mode} cell {args.get('cell_index', '?')} in {args.get('path', '?')}\n"
                f"{_diff_preview('', args.get('new_source', ''))}")
    return f"{name}({json.dumps(args)})"


def _type_feedback():
    """Prompt for a freeform instruction at an approval gate.

    The returned text becomes the tool's denial reason, so it is fed straight
    back to the model SYM_EMDASH Claude-Code's "No, and tell it what to do instead".
    """
    try:
        fb = input(f"{BOLD}  tell Hera what to do instead: {R}").strip()
    except (EOFError, KeyboardInterrupt):
        return "user declined to run this tool"
    return fb or "user declined to run this tool"


def approve(name, args):
    """Return True to run the tool, or a string denial reason."""
    # Plan mode: block anything that changes state until the user approves.
    if PLAN_MODE and name in SIDE_EFFECTS:
        return ("plan mode is on SYM_EMDASH investigate read-only, then call exit_plan_mode with your "
                "plan for the user to approve. Modifying tools stay disabled until then.")

    # Fine-grained permission rules (from config) take precedence over YOLO/allowlist.
    decision = _perm_decision(name, args)
    if decision == "deny":
        return "denied by a permission rule"

    # PreToolUse hooks may veto any tool call.
    blocked = _run_hooks("PreToolUse", name, args)
    if blocked:
        return blocked

    if decision == "allow":
        return True

    # Auto mode (per-project) and a 'ask' rule both modulate the prompt.
    if decision != "ask":
        if YOLO or AUTO_MODE == "all":
            return True
        if AUTO_MODE == "edit" and name in _EDIT_TOOLS:
            return True
        if name not in SIDE_EFFECTS or name in _always_ok:
            return True

    # run_bash: consult the command allowlist first.
    if name == "run_bash":
        cmd = args.get("command", "")
        if decision != "ask" and bash_allowed(cmd):
            if not _is_quiet():
                print(f"  {DIM}{SYM_HOOKED} auto-approved (allowlist){R}")
            return True

    # A quiet parallel sub-agent (worker thread) can't prompt — deny anything
    # not already auto-granted above, so it stays read-only unless YOLO/auto.
    if _is_quiet():
        return (f"requires approval ({name}) {SYM_EMDASH} a parallel sub-agent can't prompt; "
                f"run with --yolo or auto mode to allow it")
        denied = _matches(cmd, DENY_PATTERNS)
        print(f"\n{YELL}{BOLD}{SYM_WARN} approval needed{R} {DIM}(run_bash{', matches deny-pattern' if denied else ''}){R}")
        print(f"  $ {cmd}")
        print(f"  {DIM}sandbox: {sandbox_label()}{R}")
        _run_hooks("Notification", name=name, args=args)
        try:
            ans = input(f"{BOLD}  [y]es once / [a]lways this command / "
                        f"[p]rogram (all '{cmd.split()[0] if cmd.split() else '?'}') / "
                        f"[t]ype feedback / [n]o:{R} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "user aborted (no input)"
        if ans in ("y", "yes", ""):
            return True
        if ans in ("a", "always"):
            ALLOW_PATTERNS.append(" ".join(cmd.split()))
            return True
        if ans in ("p", "program") and cmd.split():
            prog = cmd.split()[0]
            ALLOW_PATTERNS.append(f"{prog} *")
            ALLOW_PATTERNS.append(prog)
            return True
        if ans in ("t", "type"):
            return _type_feedback()
        return "user declined to run this tool"

    # write_file / edit_file: tool-level approval.
    print(f"\n{YELL}{BOLD}{SYM_WARN} approval needed{R} {DIM}({name}){R}")
    for ln in _preview_call(name, args).split("\n"):
        print(f"  {ln}")
    _run_hooks("Notification", name=name, args=args)
    try:
        ans = input(f"{BOLD}  run this? [y]es / [a]lways / [t]ype feedback / [n]o:{R} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "user aborted (no input)"
    if ans in ("y", "yes", ""):
        return True
    if ans in ("a", "always"):
        _always_ok.add(name)
        return True
    if ans in ("t", "type"):
        return _type_feedback()
    return "user declined to run this tool"


# ── ESC-to-interrupt ──────────────────────────────────────────────────────────
@contextlib.contextmanager
def interruptible():
    """While the body runs, watch the terminal for ESC and set _INTERRUPT.

    A daemon thread puts stdin in cbreak mode and polls for the ESC byte
    (0x1b). stream_turn() checks _INTERRUPT each chunk and stops the turn.
    No-ops off a real TTY (e.g. piped input, Windows). Ctrl-C is unaffected SYM_EMDASH
    it still raises KeyboardInterrupt through the normal path.
    """
    _INTERRUPT.clear()
    if termios is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    stop = threading.Event()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        yield
        return

    def watch():
        try:
            tty.setcbreak(fd)
            while not stop.is_set():
                r, _, _ = select.select([fd], [], [], 0.1)
                if not r:
                    continue
                b = os.read(fd, 1)
                if b == b"\x1b":          # ESC
                    _INTERRUPT.set()
                    return
        except Exception:                 # noqa: BLE001 SYM_EMDASH never crash the turn
            pass

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=0.3)
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass


# ── Streaming chat call ───────────────────────────────────────────────────────
def _is_context_overflow(exc):
    """True if an HTTP 400 is the server rejecting an over-long prompt."""
    try:
        body = (exc.response.text if exc.response is not None else "") or ""
    except Exception:  # noqa: BLE001
        body = ""
    body = body.lower()
    return ("context" in body and ("exceed" in body or "larger than" in body
                                   or "too long" in body or "available context" in body))


def _scan_json_objects(s):
    """Yield (start, end, parsed_dict) for every balanced top-level {...} in s.

    String- and escape-aware brace matching, so nested objects and braces inside
    quoted strings don't fool it (a plain regex would). Used to recover tool
    calls a server left embedded in text content."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth, j, instr, esc = 0, i, False, False
        while j < n:
            c = s[j]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
            elif c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = s[i:j + 1]
                    try:
                        out.append((i, j + 1, json.loads(blob)))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
            j += 1
        i = j + 1
    return out


def _extract_text_tool_calls(content):
    """Recover tool calls a server leaked into text instead of returning as
    structured `tool_calls`.

    Some OpenAI-compatible servers (notably llama.cpp without the right tool
    template) let a model's tool call fall through as plain content SYM_EMDASH e.g. a
    Hermes/Qwen `<tool_call>{SYM_ELLIPSIS}</tool_call>` block or a fenced JSON object. We
    parse those back out so the agent still acts on them instead of treating the
    turn as a final answer. Only objects naming a *registered* tool are accepted,
    so an illustrative JSON snippet in prose isn't mistaken for a call.

    Returns (cleaned_content, calls) where calls is a list of
    {id, name, arguments(str)} mirroring the streamed shape.
    """
    if not content or "{" not in content:
        return content, []
    calls, cleaned = [], content
    for start, end, obj in _scan_json_objects(content):
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
        name = obj.get("name") or obj.get("tool") or (fn.get("name") if fn else None)
        args = obj.get("arguments")
        if args is None and fn:
            args = fn.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            args = obj.get("args")
        # Accept only real tool calls: a known tool name AND an arguments-shaped
        # field (so {"name": "x"} mentioned in prose isn't hijacked).
        if not name or name not in TOOLS or args is None:
            continue
        args_str = args if isinstance(args, str) else json.dumps(args)
        calls.append({"id": f"txt-{len(calls)}-{abs(hash(content[start:end])) % 100000}",
                      "name": name, "arguments": args_str})
        cleaned = cleaned.replace(content[start:end], "", 1)
    if calls:  # tidy any wrapper tags / empty fences left behind
        cleaned = re.sub(r"</?tool_call>", "", cleaned)
        cleaned = re.sub(r"```(?:json|tool_call|tool_code|tool)?\s*```", "", cleaned)
    return cleaned.strip(), calls


# ── Anthropic-native provider (Messages API; Bedrock/Vertex are auth variants) ─
def _is_anthropic():
    return PROVIDER in ("anthropic", "bedrock", "vertex")


def _anthropic_tools(schemas):
    """Convert OpenAI function schemas to Anthropic tool definitions."""
    out = []
    for s in schemas:
        f = s.get("function", {})
        out.append({"name": f["name"], "description": f.get("description", ""),
                    "input_schema": f.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _to_anthropic_messages(messages):
    """Translate OpenAI-style history SYM_ARROW_R (system_str, anthropic_messages).

    assistant tool_calls SYM_ARROW_R tool_use blocks; role:tool SYM_ARROW_R user tool_result blocks;
    multimodal image_url parts SYM_ARROW_R Anthropic image blocks. Consecutive same-role
    messages are merged so roles alternate as the API expects."""
    system_parts, out = [], []
    for m in messages:
        role = m.get("role")
        if role == "system":
            t = _text_of(m.get("content"))
            if t:
                system_parts.append(t)
            continue
        if role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                 "content": _text_of(m.get("content"))}]})
            continue
        if role == "assistant":
            blocks = []
            txt = _text_of(m.get("content"))
            if txt:
                blocks.append({"type": "text", "text": txt})
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                               "name": fn.get("name", ""), "input": inp})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            continue
        # user
        c = m.get("content")
        if isinstance(c, list):
            blocks = []
            for part in c:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        meta, b64 = url.split(",", 1)
                        mime = meta[5:].split(";")[0] or "image/png"
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": mime, "data": b64}})
            out.append({"role": "user", "content": blocks or [{"type": "text", "text": ""}]})
        else:
            out.append({"role": "user", "content": c or ""})
    # Merge consecutive same-role messages (Anthropic wants alternating turns).
    merged = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            for blk in (prev, msg):
                if isinstance(blk["content"], str):
                    blk["content"] = [{"type": "text", "text": blk["content"]}]
            prev["content"] = prev["content"] + msg["content"]
        else:
            merged.append(dict(msg))
    return "\n\n".join(system_parts), merged


def _parse_anthropic_response(data):
    """Convert an Anthropic Messages response dict SYM_ARROW_R internal turn shape."""
    parts, tool_calls = [], []
    for block in data.get("content", []) or []:
        bt = block.get("type")
        if bt == "text":
            parts.append(block.get("text", ""))
        elif bt == "tool_use":
            tool_calls.append({"id": block.get("id", ""), "name": block.get("name", ""),
                               "arguments": json.dumps(block.get("input", {}))})
    u = data.get("usage", {}) or {}
    pin, pout = u.get("input_tokens", 0), u.get("output_tokens", 0)
    usage = {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}
    fr = "tool_calls" if tool_calls else (data.get("stop_reason") or "stop")
    return {"content": "".join(parts), "tool_calls": tool_calls,
            "finish_reason": fr, "usage": usage}


def _anthropic_headers():
    """Auth + base URL for the active Anthropic-family provider."""
    if PROVIDER == "vertex":            # GCP: OAuth bearer access token in API_KEY
        return ANTHROPIC_BASE, {"Authorization": f"Bearer {API_KEY}",
                                "Content-Type": "application/json"}
    if PROVIDER == "bedrock":           # AWS: expects SigV4 (front with a proxy) or a bearer
        return ANTHROPIC_BASE, {"Authorization": f"Bearer {API_KEY}",
                                "Content-Type": "application/json"}
    return ANTHROPIC_BASE, {"x-api-key": API_KEY, "anthropic-version": ANTHROPIC_VERSION,
                            "Content-Type": "application/json"}


def _anthropic_turn(messages, spinner, tools=None, model_override=None):
    """One turn against the Anthropic Messages API. Non-streaming for robustness;
    prints the answer once it arrives. Returns the internal turn dict or None."""
    base, headers = _anthropic_headers()
    system, amsgs = _to_anthropic_messages(messages)
    schemas = tools if tools is not None else TOOL_SCHEMAS
    body = {"model": model_override or MODEL, "max_tokens": MAX_OUTPUT_TOKENS, "messages": amsgs,
            "tools": _anthropic_tools(schemas)}
    if system:
        body["system"] = system
    try:
        resp = requests.post(f"{base}/v1/messages", json=body, headers=headers, timeout=600)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        spinner.stop()
        print(f"{RED}[error] {exc}{R}\n", file=sys.stderr)
        return None
    result = _parse_anthropic_response(data)
    spinner.stop()
    if result["content"]:
        print(f"\n{ACCENT}{SYM_HALF_L}{R} {BOLD}{NAME}{R}\n")
        md_state = {"code": False}
        for ln in result["content"].split("\n"):
            print(render_md_line(ln, md_state), flush=True)
    return result


def stream_turn(messages, spinner, tools=None, model_override=None):
    """One model turn. Streams reasoning + content live; assembles tool calls.
    `model_override` (e.g. from a sub-agent definition) selects a different model.

    Returns dict {content, finish_reason, tool_calls, usage} or None on error.
    """
    if _is_anthropic():
        return _anthropic_turn(messages, spinner, tools, model_override)
    # Transient blips (server busy under --parallel load, a dropped connection)
    # return 5xx/connection errors. Retry a few times with backoff before giving
    # up, so one hiccup doesn't end the turn.
    global _VISION_WARNED
    url, model, send_messages, downgraded = _select_endpoint(messages)
    if model_override:
        model = model_override
    if downgraded and not _VISION_WARNED:
        _VISION_WARNED = True
        spinner.stop()
        print(f"{YELL}{SYM_WARN} image attached, but the current model is text-only {SYM_EMDASH} "
              f"set HERA_VISION_URL to enable vision.{R}", file=sys.stderr)

    resp = None
    last_err = None
    compacted = False
    attempt = 0
    while attempt < 3:
        try:
            resp = requests.post(
                f"{url}/chat/completions",
                json={
                    "model": model,
                    "messages": send_messages,
                    "tools": tools if tools is not None else TOOL_SCHEMAS,
                    "tool_choice": "auto",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **_think_payload(),
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                stream=True,
                timeout=600,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as exc:
            last_err = exc
            code = exc.response.status_code if exc.response is not None else 0
            # Self-heal a context-overflow 400: summarize the history and retry
            # once, so a big file read mid-task can't dead-end the turn.
            if (code == 400 and _is_context_overflow(exc)
                    and not compacted and len(messages) > 2):
                spinner.stop()
                print(f"{DIM}  context window full {SYM_EMDASH} compacting history and retrying{SYM_ELLIPSIS}{R}",
                      file=sys.stderr)
                compact_history(messages)
                url, model, send_messages, downgraded = _select_endpoint(messages)
                if model_override:
                    model = model_override
                compacted = True
                spinner.start()
                continue  # doesn't count as a failed attempt
            if code and code < 500:
                spinner.stop()  # other 4xx (401/403) won't fix itself SYM_EMDASH don't retry
                print(f"{RED}[error] {exc}{R}\n", file=sys.stderr)
                return None
        except requests.exceptions.ConnectionError as exc:
            last_err = exc
        except requests.exceptions.Timeout as exc:
            last_err = exc
        attempt += 1
        if attempt < 3:
            time.sleep(1.5 * attempt)  # 1.5s, 3s

    if resp is None:
        spinner.stop()
        if isinstance(last_err, requests.exceptions.ConnectionError):
            print(f"{RED}[error] Cannot reach {url}{R}\n", file=sys.stderr)
        elif isinstance(last_err, requests.exceptions.Timeout):
            print(f"{RED}[error] Request timed out{R}\n", file=sys.stderr)
        else:
            print(f"{RED}[error] {last_err} (server kept failing after retries){R}\n",
                  file=sys.stderr)
        return None

    content       = []
    tool_calls    = {}     # index -> {id, name, arguments}
    finish_reason = None
    usage         = None
    started       = False
    in_reasoning  = False

    line_buf = ""               # buffers content until a newline, then renders it
    md_state = {"code": False}

    quiet = _is_quiet()

    def ensure_header():
        nonlocal started
        if not started:
            started = True
            if quiet:
                return
            elapsed = spinner.stop()
            print(f"\n{ACCENT}{SYM_HALF_L}{R} {BOLD}{NAME}{R}  {GREY}{SYM_MIDDOT} {elapsed:.1f}s to first token{R}"
                  f"  {DIM}(esc to interrupt){R}\n")

    for raw in resp.iter_lines():
        if _INTERRUPT.is_set():
            ensure_header()
            print(f"\n{R}{DIM}{SYM_INTERRUPT} (interrupted by ESC){R}", flush=True)
            finish_reason = "interrupted"
            resp.close()
            break
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue  # e.g. the final usage-only chunk
        choice = choices[0]
        delta = choice.get("delta", {})
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        reasoning = delta.get("reasoning_content", "")
        token     = delta.get("content", "")

        if reasoning and not HIDE_REASONING and not quiet:
            ensure_header()
            if not in_reasoning:
                print(f"{DIM}{ITAL}{SYM_THINK} thinking{SYM_ELLIPSIS}{R}\n{DIM}", end="", flush=True)
                in_reasoning = True
            print(f"{DIM}{reasoning}{R}", end="", flush=True)

        if token:
            content.append(token)
            if not quiet:
                ensure_header()
                if in_reasoning:
                    print(f"{R}\n\n", end="", flush=True)
                    in_reasoning = False
                line_buf += token
                while "\n" in line_buf:
                    done, line_buf = line_buf.split("\n", 1)
                    print(render_md_line(done, md_state), flush=True)

        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function", {})
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    if in_reasoning and not quiet:
        print(f"{R}\n", end="", flush=True)
    if line_buf and not quiet:  # flush any trailing partial line
        print(render_md_line(line_buf, md_state), flush=True)

    final_content = "".join(content)
    structured = [tool_calls[i] for i in sorted(tool_calls)]
    # Fallback: if the server returned no structured tool calls but left them in
    # the text (a misconfigured tool template), recover them so the agent acts.
    if not structured:
        final_content, recovered = _extract_text_tool_calls(final_content)
        if recovered:
            structured = recovered
            if not quiet:
                print(f"{DIM}  {SYM_HOOKED} recovered {len(recovered)} tool call(s) from text "
                      f"(server didn't structure them){R}", flush=True)

    return {
        "content": final_content,
        "finish_reason": finish_reason,
        "tool_calls": structured,
        "usage": usage,
    }


def _account(usage):
    """Fold a request's usage into the session totals; return per-request total."""
    if not usage:
        return 0
    global _LAST_PROMPT_TOKENS
    p = usage.get("prompt_tokens", 0)
    c = usage.get("completion_tokens", 0)
    t = usage.get("total_tokens", p + c)
    # Remember the server's real prompt size so auto-compaction triggers on
    # ground truth rather than a rough char estimate.
    _LAST_PROMPT_TOKENS = p + c
    SESSION["prompt"]     += p
    SESSION["completion"] += c
    SESSION["total"]      += t
    SESSION["requests"]   += 1
    _emit_telemetry("request", prompt_tokens=p, completion_tokens=c, total_tokens=t)
    _emit_metric("hera.tokens", t, kind="total")
    _emit_metric("hera.requests", 1)
    return t


# ── Edit checkpoints / undo ───────────────────────────────────────────────────
CHECKPOINTS = []  # stack of {path, existed, content, label}


def _snapshot(path):
    """Capture a file's prior state: (existed, content-or-None)."""
    p = _resolve(path)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return True, f.read()
        except OSError:
            return True, None
    return False, None


def push_checkpoint(path, snap, label):
    CHECKPOINTS.append({"path": _resolve(path), "existed": snap[0],
                        "content": snap[1], "label": label})


def undo_last():
    if not CHECKPOINTS:
        return "nothing to undo"
    cp = CHECKPOINTS.pop()
    p = cp["path"]
    try:
        if cp["existed"]:
            with open(p, "w", encoding="utf-8") as f:
                f.write(cp["content"] or "")
            return f"reverted {p} (undid {cp['label']})"
        if os.path.exists(p):
            os.remove(p)
        return f"deleted {p} (it was created by {cp['label']})"
    except OSError as exc:
        return f"[error] undo failed: {exc}"


# ── Rewind: restore conversation + code to an earlier turn (Claude-Code style) ─
# Each user turn records where the message history and the edit-checkpoint stack
# stood when it began, so /rewind can roll BOTH back to that point at once.
TURN_MARKS = []  # [{"msg": int, "ckpt": int, "text": str}]


def mark_turn(messages, text):
    TURN_MARKS.append({"msg": len(messages), "ckpt": len(CHECKPOINTS),
                       "text": (_text_of(text) or "").strip()})


def reset_turn_marks():
    TURN_MARKS.clear()


def rewind_to(messages, mark):
    """Roll the conversation and the files back to the state at `mark`. Returns
    the number of file edits reverted."""
    reverted = 0
    while len(CHECKPOINTS) > mark["ckpt"]:
        undo_last()
        reverted += 1
    del messages[mark["msg"]:]
    while TURN_MARKS and TURN_MARKS[-1]["msg"] >= mark["msg"]:
        TURN_MARKS.pop()
    return reverted


def rewind_picker(messages, n=None):
    """Interactive /rewind: pick an earlier user turn to restore to. With `n`,
    jump straight back n turns."""
    if not TURN_MARKS:
        print(f"\n{DIM}nothing to rewind to yet.{R}\n")
        return
    if n is not None:
        idx = max(0, len(TURN_MARKS) - n)
        mark = TURN_MARKS[idx]
        cnt = rewind_to(messages, mark)
        print(f"\n{DIM}rewound {n} turn(s) {SYM_EMDASH} {cnt} file edit(s) reverted.{R}\n")
        save_session(messages)
        return
    shown = TURN_MARKS[-9:]
    base = len(TURN_MARKS) - len(shown)
    print(f"\n{DIM}rewind to before which message? (restores conversation AND files){R}")
    for i, m in enumerate(shown, 1):
        preview = (m["text"][:70] or "(empty)").replace("\n", " ")
        print(f"  {CYAN}{i}{R} {DIM}{preview}{R}")
    try:
        sel = input(f"{BOLD}  number (Enter to cancel): {R}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not sel.isdigit() or not (1 <= int(sel) <= len(shown)):
        print(f"{DIM}cancelled.{R}\n")
        return
    mark = shown[int(sel) - 1]
    cnt = rewind_to(messages, mark)
    print(f"\n{DIM}rewound to before {(mark['text'][:50] or '(empty)')!r} {SYM_EMDASH} "
          f"{cnt} file edit(s) reverted, conversation truncated.{R}\n")
    save_session(messages)


def context_report(messages):
    """A Claude-Code-style /context breakdown: where the window is going."""
    win = CONTEXT_TOKENS or 131072
    sys_tok = len(_text_of(messages[0].get("content"))) // 4 if messages else 0
    convo_tok = _estimate_tokens(messages[1:]) if len(messages) > 1 else 0
    todo_tok = len(json.dumps(TODOS)) // 4 if TODOS else 0
    est = sys_tok + convo_tok + todo_tok
    # Prefer the server's real last prompt size when we have it.
    real = _LAST_PROMPT_TOKENS or est
    pct = min(100, round(100 * real / win)) if win else 0
    filled = int(pct / 5)
    bar = SYM_PROG_F * filled + SYM_PROG_E * (20 - filled)
    n_user = sum(1 for m in messages if m.get("role") == "user")
    n_tool = sum(1 for m in messages if m.get("role") == "tool")
    lines = [
        f"\n{BOLD}Context window{R}  {DIM}({SYM_APPROX} {real:,} / {win:,} tokens){R}",
        f"  {bar} {pct}%",
        f"{DIM}  system prompt + project context : ~{sys_tok:,} tok",
        f"  conversation ({len(messages)} msgs: {n_user} user, {n_tool} tool) : ~{convo_tok:,} tok",
        f"  to-dos : ~{todo_tok:,} tok",
        f"  auto-compaction triggers at {int((AUTO_COMPACT_AT or 0) * 100)}% of the window{R}",
    ]
    if pct >= int((AUTO_COMPACT_AT or 0.8) * 100):
        lines.append(f"{YELL}  {SYM_WARN} near the limit {SYM_EMDASH} /compact to summarize and free space.{R}")
    print("\n".join(lines) + "\n")


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(messages, spinner):
    """Drive the reasonSYM_ARROW_Ract loop until the model produces a final answer."""
    turn_tokens = 0
    did_work = False       # did this turn actually use tools? (gates next-step tips)
    edited_code = False    # wrote/edited a code file
    ran_command = False    # ran a shell command (SYM_APPROX self-verified)
    verified = False       # already injected the auto-verify nudge for this task
    edit_attempted = False # the model emitted at least one edit/write call
    edit_nudged = False    # already injected the apply-your-edit nudge for this task
    last_user = next((_text_of(m.get("content")) for m in reversed(messages)
                      if m.get("role") == "user"), "")
    # Did the user ask to run/verify an (existing) project? Then verify even
    # without edits — like "run this", "make sure it works", "run the tests".
    verify_requested = _wants_run_verification(last_user)
    # Did the user ask for a code change? Then if the model only *talks* about
    # the fix and never edits, nudge it once to actually apply it.
    change_requested = _wants_code_change(last_user)
    globals()["_TURN_THINK"] = _keyword_think_level(last_user)
    _maybe_auto_compact(messages)
    step = 0
    while True:
        if MAX_STEPS and step >= MAX_STEPS:
            print(f"\n{RED}[stopped] hit MAX_STEPS={MAX_STEPS} tool round-trips{R}\n")
            return True
        step += 1
        spinner.start()
        # Only the streaming call watches for ESC — tool approvals use input()
        # on the same stdin, so the watcher must be off while they run.
        with interruptible():
            result = stream_turn(messages, spinner)
        if result is None:
            return False  # transport error; caller rolls back the user message

        turn_tokens += _account(result.get("usage"))

        # ESC mid-stream: keep the partial answer (as a clean string message so
        # history isn't poisoned by dangling tool_calls) and hand control back.
        if result.get("finish_reason") == "interrupted":
            messages.append({"role": "assistant",
                             "content": result["content"] or "(interrupted)"})
            print(f"{GREY}  {turn_tokens} tok this turn {SYM_MIDDOT} {SESSION['total']} session"
                  f"{_cost_suffix()}  {SYM_MIDDOT} stopped by ESC{R}\n")
            return True

        calls = result["tool_calls"]
        # Defensive: a turn that emits only tool calls (no streamed content/
        # reasoning) never trips ensure_header(), so stop the spinner here
        # before the tool cards and their own spinners print.
        spinner.stop()

        assistant_msg = {"role": "assistant", "content": result["content"] or ""}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": _normalize_tool_args(c["arguments"])}}
                for c in calls
            ]
        messages.append(assistant_msg)

        if not calls:
            # Apply-your-edit loop: the user asked for a change but the model
            # ended its turn without ever calling an edit/write tool (a local
            # model often narrates the patch instead of emitting the call).
            # Nudge once to actually apply it before declaring the task done.
            if (AUTO_VERIFY and change_requested and not edit_attempted
                    and not edit_nudged and not PLAN_MODE):
                edit_nudged = True
                messages.append({"role": "user", "content": _EDIT_NUDGE})
                print(f"\n{DIM}  {SYM_RECYCLE} auto-apply: you described a change but didn't make "
                      f"it {SYM_EMDASH} applying now (ESC to skip){SYM_ELLIPSIS}{R}")
                continue
            # Verify-your-work loop: if code was written (or the user asked to run
            # the project) but nothing was run, nudge once to actually run it and
            # fix failures before declaring the task done.
            if (AUTO_VERIFY and (edited_code or verify_requested) and not ran_command
                    and not verified and not PLAN_MODE):
                verified = True
                edited_code = False
                messages.append({"role": "user", "content": _verify_nudge()})
                print(f"\n{DIM}  {SYM_RECYCLE} auto-verify: making sure it actually runs "
                      f"(ESC to skip){SYM_ELLIPSIS}{R}")
                continue
            print(f"\n{GREY}{SYM_HLINE * 50}{R}")
            print(f"{GREY}  {turn_tokens} tok this turn {SYM_MIDDOT} {SESSION['total']} session"
                  f"{_cost_suffix()}{R}\n")
            _run_hooks("Stop")
            if did_work:
                _suggest_next_steps(messages)
            return True

        did_work = True
        for c in calls:
            output = _exec_call(c)
            if c["name"] == "run_bash":
                ran_command = True
            elif c["name"] in _EDIT_TOOLS:
                edit_attempted = True
                try:
                    cargs = json.loads(c["arguments"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    cargs = {}
                if _is_code_file(cargs.get("path", "")):
                    edited_code = True
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": output})


def _parse_suggestions(raw):
    """Pull a list of next-step strings out of the model's reply (JSON array
    preferred, bullet/numbered lines as a fallback)."""
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw or "")
    raw = re.sub(r"(?is)```(?:json)?", "", raw).strip()
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip().rstrip(".") for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    out = []
    for line in raw.splitlines():
        line = re.sub(r"^[-*\d.)\s]+", "", line.strip()).strip()
        if line:
            out.append(line.rstrip("."))
    return out


def _generate_suggestions(messages):
    """Ask the model for 2-3 concrete next steps based on the just-finished task.

    Returns a list of short strings (possibly empty). Best-effort: returns [] on
    any error or when disabled with HERA_NO_SUGGESTIONS=1. Shared by the
    interactive CLI and the --serve path (VS Code), so it neither prints nor
    touches the TTY.
    """
    if _truthy(_env("HERA_NO_SUGGESTIONS")):
        return []
    recent = []
    for m in messages[-10:]:
        role = m.get("role")
        if role == "system":
            continue
        c = _text_of(m.get("content"))
        if m.get("tool_calls"):
            c += " [ran: " + ", ".join(tc["function"]["name"] for tc in m["tool_calls"]) + "]"
        if c.strip():
            recent.append(f"{role}: {c[:600]}")
    if not recent:
        return []
    prompt = (
        "You are a terminal coding assistant that just finished a task. From the session "
        "transcript below, propose 2-3 concrete, genuinely useful next steps the user could "
        "take now (e.g. run the tests, review the diff, commit, try an edge case). Each must be "
        "a short imperative phrase of at most 9 words, specific to what was just done. Reply "
        "with ONLY a JSON array of strings SYM_EMDASH no prose, no numbering.\n\n" + "\n".join(recent)
    )
    try:
        resp = requests.post(
            f"{API_URL}/chat/completions",
            # Disable the model's thinking for this tiny call: otherwise a
            # reasoning model spends the whole token budget on hidden reasoning
            # and returns empty content. enable_thinking=false → fast, direct.
            json={"model": MODEL, "stream": False, "max_tokens": 200,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"].get("content") or ""
    except Exception:  # noqa: BLE001 SYM_EMDASH suggestions are a nicety, never fatal
        return []
    return _parse_suggestions(raw)[:3]


def _suggest_next_steps(messages):
    """Interactive (TTY) end-of-task next-step tips SYM_EMDASH like Claude Code's. Quiet:
    TTY only; the disable/error handling lives in _generate_suggestions."""
    if not sys.stdout.isatty():
        return
    spin = Spinner()
    spin.start(label="figuring out next steps")
    try:
        steps = _generate_suggestions(messages)
    finally:
        spin.stop()
    if not steps:
        return
    print(f"{DIM}  Next steps{R}")
    for s in steps:
        print(f"  {ACCENT}{SYM_SUBARROW}{R} {s}")
    print()


def _normalize_tool_args(raw):
    """Return a JSON-object string that is always safe to store in the history.

    The model occasionally streams empty or malformed `arguments` (e.g. a
    write_file call with no body). Storing that raw string poisons the
    conversation: llama.cpp's chat template can't re-render it, so *every*
    later request 500s and the session is wedged forever. Coerce to valid
    JSON SYM_EMDASH keep it if it parses to an object, otherwise drop to `{}`.
    """
    try:
        parsed = json.loads(raw or "{}")
        if isinstance(parsed, dict):
            return json.dumps(parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    return "{}"


def _sanitize_history(messages):
    """Heal a (possibly already-poisoned) message list in place.

    Normalizes every assistant tool_call's arguments so a session saved before
    this fix SYM_EMDASH or resumed from disk SYM_EMDASH can't keep 500-ing on load.
    """
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function")
            if isinstance(fn, dict):
                fn["arguments"] = _normalize_tool_args(fn.get("arguments"))
    return messages


def _exec_call(c, indent=""):
    """Execute one tool call: print, approve, checkpoint, run. Returns output string."""
    name = c["name"]
    quiet = _is_quiet()  # parallel sub-agent: suppress interleaving live output
    try:
        args = json.loads(c["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}

    # Announce the action in plain language first, then the tool card, so a
    # reader can follow what Hera is doing without parsing tool internals.
    if not quiet:
        print(f"\n{indent}{ACCENT}{SYM_ARROW_R}{R} {_narrate(name, args)}")
        print(f"{indent}{TEAL}{SYM_DIAMOND}{R} {BOLD}{name}{R}  "
              f"{GREY}{_preview_call(name, args).splitlines()[0]}{R}")

    if name not in TOOLS:
        if not quiet:
            print(f"{indent}  {GREY}{SYM_INTERRUPT}{R} {RED}unknown tool{R}")
        return f"[error] unknown tool: {name}"

    verdict = approve(name, args)
    if verdict is not True:
        if not quiet:
            print(f"{indent}  {GREY}{SYM_INTERRUPT}{R} {RED}{SYM_CROSS} {verdict}{R}")
        return f"[denied] {verdict}"

    # Snapshot file state before a mutating edit so /undo can revert it.
    snap = _snapshot(args.get("path", "")) if name in _EDIT_TOOLS else None
    # Show a live whimsy spinner while the tool runs in the "background" — fast
    # tools (read_file, list_dir) finish before it's noticeable; slow ones
    # (run_bash, web_search, web_fetch, task) get visible activity. Skip it for
    # tools that prompt the user themselves (exit_plan_mode), so the spinner
    # doesn't fight their input prompt — and for quiet parallel sub-agents.
    tspin = None if (name == "exit_plan_mode" or quiet) else Spinner()
    if tspin:
        tspin.start()
    try:
        output = TOOLS[name](**args)
    except TypeError as exc:
        msg = str(exc).replace(f"tool_{name}()", f"{name}()")
        output = f"[error] bad arguments: {msg}"
    except Exception as exc:  # noqa: BLE001 SYM_EMDASH surface to the model
        output = f"[error] {type(exc).__name__}: {exc}"
    finally:
        if tspin:
            tspin.stop()
    if snap is not None and not str(output).startswith("[error]"):
        push_checkpoint(args.get("path", ""), snap, name)
    preview = (output.splitlines()[0] if output else "(no output)")[:100]
    is_err = str(output).startswith("[error]")
    if not quiet:
        print(f"{indent}  {GREY}{SYM_INTERRUPT}{R} {(RED if is_err else GREY)}{preview}{R}")
    # Show the refreshed checklist right after a to-do update.
    if name == "todo_write" and TODOS and not is_err and not quiet:
        print(_render_todos_text())
    _run_hooks("PostToolUse", name, args, output)
    _emit_telemetry("tool", tool=name, error=is_err)
    _emit_metric("hera.tool.calls", 1, tool=name, error=is_err)

    if MAX_TOOL_OUTPUT and len(output) > MAX_TOOL_OUTPUT:
        output = output[:MAX_TOOL_OUTPUT] + f"\n{SYM_ELLIPSIS}[truncated, {len(output)} chars total]"
    return output


# ── Sub-agents / task delegation ──────────────────────────────────────────────
def _agent_model(agent):
    """The model a named agent prefers (its `model:` frontmatter), or None."""
    if agent and agent in CUSTOM_AGENTS:
        meta, _ = _parse_frontmatter(CUSTOM_AGENTS[agent])
        m = (meta.get("model") or "").strip()
        return m or None
    return None


def run_subagent(description, agent=None, model=None, quiet=False):
    """Run a focused nested agent on `description`; return its final answer text.

    The sub-agent has every tool except `task` itself (so it can't recurse),
    shares the approval gate / sandbox / checkpoints, and reports indented
    progress. Only its final summary goes back to the parent. A named agent
    definition (from ~/.config/hera/agents/<name>.md) can supply its own system
    prompt, restrict the tool set via a `tools:` frontmatter list, and pick a
    different `model:` (overridable per call via the task tool's `model` arg).

    `quiet=True` runs it on the calling worker thread without live output (used
    by parallel delegation) SYM_EMDASH its result is still returned to the parent.
    """
    if quiet:
        _QUIET.on = True
    try:
        sub_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] != "task"]
        sys_prompt = (f"You are a focused sub-agent of {NAME}. Complete the delegated task using "
                      f"your tools in {os.getcwd()}, then reply with a concise summary of what you "
                      f"found or changed. Be thorough but terse.")
        label = "sub-agent"
        sub_model = model or _agent_model(agent)
        if agent and agent in CUSTOM_AGENTS:
            meta, body = _parse_frontmatter(CUSTOM_AGENTS[agent])
            if body.strip():
                sys_prompt = body.strip() + f"\n\nWork in {os.getcwd()}. Reply with a concise summary."
            label = f"agent:{agent}"
            allowed = [t.strip() for t in (meta.get("tools", "")).replace(",", " ").split()
                       if t.strip()]
            if allowed:
                sub_schemas = [s for s in sub_schemas if s["function"]["name"] in allowed]
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": description},
        ]
        spinner = Spinner()
        tag = f"{label}" + (f" ({sub_model})" if sub_model else "")
        if not quiet:
            print(f"\n  {BLUE}{SYM_SUB} {tag} started{R} {DIM}{description[:70]}{R}")
        final = ""
        sub_step = 0
        while True:
            if MAX_STEPS and sub_step >= MAX_STEPS:
                break
            sub_step += 1
            if not quiet:
                spinner.start()
            res = stream_turn(msgs, spinner, tools=sub_schemas, model_override=sub_model)
            if res is None:
                return "[sub-agent error] transport failure"
            _account(res.get("usage"))
            calls = res["tool_calls"]
            am = {"role": "assistant", "content": res["content"] or ""}
            if calls:
                am["tool_calls"] = [{"id": c["id"], "type": "function",
                                     "function": {"name": c["name"],
                                                  "arguments": _normalize_tool_args(c["arguments"])}}
                                    for c in calls]
            msgs.append(am)
            if not calls:
                final = res["content"] or ""
                break
            for c in calls:
                out = _exec_call(c, indent="    ")
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        if not quiet:
            print(f"  {BLUE}{SYM_SUB} {label} done{R}")
        _run_hooks("SubagentStop", name=label, output=final)
        return final or "(sub-agent produced no result)"
    finally:
        if quiet:
            _QUIET.on = False


def run_subagents_parallel(specs):
    """Run several sub-agents concurrently (each quiet, on its own thread) and
    return their combined results. `specs` is a list of {description, agent?, model?}."""
    results = [None] * len(specs)

    def worker(i, spec):
        results[i] = run_subagent(spec.get("description", ""), agent=spec.get("agent"),
                                  model=spec.get("model"), quiet=True)

    threads = []
    for i, spec in enumerate(specs):
        t = threading.Thread(target=worker, args=(i, spec), daemon=True)
        t.start()
        threads.append(t)
    print(f"\n  {BLUE}{SYM_SUB} running {len(specs)} sub-agents in parallel{SYM_ELLIPSIS}{R}")
    for t in threads:
        t.join()
    print(f"  {BLUE}{SYM_SUB} {len(specs)} sub-agents done{R}")
    return "\n\n".join(f"### sub-agent {i + 1}"
                       + (f" ({specs[i].get('agent')})" if specs[i].get('agent') else "")
                       + f"\n{results[i]}" for i in range(len(specs)))


def tool_task(description=None, agent=None, model=None, tasks=None, **_ignored):
    # `tasks` (a list) runs several sub-agents in parallel; otherwise a single one.
    if isinstance(tasks, list) and tasks:
        return run_subagents_parallel(tasks)
    return run_subagent(description or "", agent=agent, model=model)


TOOLS["task"] = tool_task
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "task",
    "description": ("Delegate a self-contained subtask to a focused sub-agent that has the "
                    "same tools and returns a concise result. Use for multi-step research or "
                    "work you want handled in one shot (e.g. 'find and summarize all places "
                    "that read config'). Optionally target a named agent (configured under "
                    "~/.config/hera/agents) with its own instructions via the `agent` field."),
    "parameters": {"type": "object", "properties": {
        "description": {"type": "string", "description": "The subtask, with enough context to act on"},
        "agent": {"type": "string", "description": "Optional named agent to use"},
        "model": {"type": "string", "description": "Optional model id to run this sub-agent on"},
        "tasks": {"type": "array", "description": ("Optional: run several sub-agents IN PARALLEL. "
                  "Each item is an object with description (and optional agent/model). Use this "
                  "for independent subtasks you want done concurrently."),
                  "items": {"type": "object", "properties": {
                      "description": {"type": "string"},
                      "agent": {"type": "string"},
                      "model": {"type": "string"}}, "required": ["description"]}},
    }},
}})


# Web tools are read-only (no approval gate) so the model can look things up on
# its own the moment it lacks information. Disable with HERA_NO_WEB=1.
if WEB_ENABLED:
    TOOLS["web_search"] = tool_web_search
    TOOLS["web_fetch"] = tool_web_fetch
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": "web_search",
        "description": ("Search the live web and return ranked results (title, URL, "
                        "snippet). Use this automatically whenever you lack the "
                        "information to answer SYM_EMDASH e.g. current events, library/API "
                        "docs, versions, error messages SYM_EMDASH instead of guessing."),
        "parameters": {"type": "object", "properties": {
            "query":       {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "How many results (default 6)"},
        }, "required": ["query"]},
    }})
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": "web_fetch",
        "description": ("Fetch a web page (by URL, e.g. one returned by web_search) "
                        "and return its readable text, so you can read docs or articles. "
                        "Fetch the top few relevant results, not just one, so you can "
                        "corroborate and synthesize before answering."),
        "parameters": {"type": "object", "properties": {
            "url":       {"type": "string", "description": "Page URL to fetch"},
            "max_chars": {"type": "integer", "description": "Max characters to return (default 9000)"},
        }, "required": ["url"]},
    }})


# install_tool lets the model proactively request a program it needs. It's a
# side-effect (downloads + installs software) so the user must approve it via
# the normal gate before anything is fetched. Disable with HERA_NO_AUTOINSTALL=1.
if AUTO_INSTALL:
    TOOLS["install_tool"] = tool_install
    SIDE_EFFECTS.add("install_tool")
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": "install_tool",
        "description": ("Install a command-line program/tool you need for the task but "
                        "that isn't available yet (e.g. jq, ripgrep, a formatter, a CLI). "
                        "The user is asked to approve before anything is downloaded; once "
                        "approved it's installed with the system package manager and you "
                        "can then use it via run_bash. Give a short reason."),
        "parameters": {"type": "object", "properties": {
            "program": {"type": "string", "description": "The command/binary name to install, e.g. 'jq'"},
            "reason":  {"type": "string", "description": "Why you need it (one short phrase)"},
        }, "required": ["program"]},
    }})


# ── Claude-parity features: cost · to-dos · hooks · permissions · plan · jobs ──

def _session_cost():
    """Estimated session cost in USD, or None when no pricing is configured."""
    if PRICE_IN <= 0 and PRICE_OUT <= 0:
        return None
    return (SESSION["prompt"] / 1e6) * PRICE_IN + (SESSION["completion"] / 1e6) * PRICE_OUT


def _cost_suffix():
    c = _session_cost()
    return f" {SYM_MIDDOT} ${c:.4f}" if c is not None else ""


# ----- to-do list (Claude-Code-style task checklist) -------------------------
_TODO_MARK = {"completed": SYM_DONE, "in_progress": SYM_INPROG, "pending": SYM_PENDING}


def tool_todo_write(todos, **_ignored):
    """Replace the working to-do list. `todos` = list of {content, status}."""
    global TODOS
    clean = []
    for t in (todos or []):
        if not isinstance(t, dict):
            continue
        content = str(t.get("content", "")).strip()
        if not content:
            continue
        status = t.get("status", "pending")
        if status not in _TODO_MARK:
            status = "pending"
        clean.append({"content": content, "status": status})
    TODOS = clean
    done = sum(1 for t in clean if t["status"] == "completed")
    return f"updated plan ({done}/{len(clean)} done)"


def _render_todos_text():
    lines = []
    for t in TODOS:
        m = _TODO_MARK.get(t["status"], SYM_PENDING)
        body = t["content"]
        if t["status"] == "completed":
            body = f"{GREY}{body}{R}"
        elif t["status"] == "in_progress":
            body = f"{BOLD}{body}{R}"
        lines.append(f"  {ACCENT}{m}{R} {body}")
    return "\n".join(lines)


TOOLS["todo_write"] = tool_todo_write
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "todo_write",
    "description": ("Maintain a visible to-do checklist for the current task. Call this at the "
                    "start of any multi-step task to lay out the steps, and again after each step "
                    "to mark it completed and set the next one in_progress. Keep exactly one item "
                    "in_progress at a time. Skip it for trivial single-step requests."),
    "parameters": {"type": "object", "properties": {
        "todos": {"type": "array", "description": "The full updated list (replaces the previous one)",
                  "items": {"type": "object", "properties": {
                      "content": {"type": "string", "description": "Short imperative step"},
                      "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                  }, "required": ["content", "status"]}},
    }, "required": ["todos"]},
}})


# ----- plan mode: present a plan and get approval to execute (Claude-Code style)
_PLAN_APPROVER = None  # set to a serve approver in --serve; terminal prompt otherwise


def _confirm_plan(plan):
    """Show the plan, return (decision, feedback) where decision is
    'yes' | 'auto' | 'no'. Pluggable so --serve can route to the editor."""
    if _PLAN_APPROVER is not None:
        return _PLAN_APPROVER(plan)
    print(f"\n{ACCENT}{SYM_HALF_L}{R} {BOLD}Ready to code?{R} {DIM}Here's the plan:{R}\n")
    for ln in (plan or "(no plan provided)").splitlines():
        print(f"  {ln}")
    try:
        ans = input(f"\n{BOLD}  [1] yes, proceed   [2] yes + auto-accept edits   "
                    f"[3] no, keep planning:{R} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "no", ""
    if ans in ("1", "y", "yes", ""):
        return "yes", ""
    if ans in ("2", "a", "auto"):
        return "auto", ""
    if ans in ("3", "n", "no"):
        return "no", ""
    return "no", ans  # any other text SYM_ARROW_R keep planning, with that as feedback


def tool_exit_plan_mode(plan="", **_ignored):
    """Present the finished plan to the user and ask whether to proceed. On
    approval, leave plan mode so the implementation can begin."""
    global PLAN_MODE, AUTO_MODE
    if not PLAN_MODE:
        return "Not in plan mode SYM_EMDASH go ahead and implement directly."
    decision, feedback = _confirm_plan(plan or "(no plan text provided)")
    if decision == "no":
        msg = ("The user is NOT ready SYM_EMDASH keep planning and do not make any changes yet.")
        if feedback:
            msg += f" Their feedback: {feedback}"
        return msg + " Revise the plan and call exit_plan_mode again when ready."
    PLAN_MODE = False
    if decision == "auto":
        AUTO_MODE = "edit"
        _save_auto_mode("edit")
        return ("Plan APPROVED and the user chose to auto-accept edits. Plan mode is OFF SYM_EMDASH "
                "implement the plan now; file edits run without individual prompts.")
    return ("Plan APPROVED by the user. Plan mode is OFF SYM_EMDASH implement the plan now with your "
            "tools (edits/commands will prompt for approval as usual).")


TOOLS["exit_plan_mode"] = tool_exit_plan_mode
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "exit_plan_mode",
    "description": ("Use ONLY in plan mode, once you've finished investigating and have a concrete "
                    "plan. Presents your plan to the user for approval; if they approve, plan mode "
                    "turns off and you implement it. Do not call it for pure question-answering."),
    "parameters": {"type": "object", "properties": {
        "plan": {"type": "string", "description": "The plan to execute, as concise markdown "
                 "(numbered steps). Shown to the user for approval."},
    }, "required": ["plan"]},
}})


# ----- background shell jobs --------------------------------------------------
def _bash_background_start(command):
    """Launch a long-running command detached; return a job id immediately."""
    global _BG_SEQ
    argv, use_shell = _sandbox_argv(command)
    try:
        out = tempfile.NamedTemporaryFile(prefix="hera-bg-", suffix=".log",
                                          delete=False, mode="w")
        proc = subprocess.Popen(argv, shell=use_shell, stdout=out, stderr=subprocess.STDOUT,
                                 cwd=os.getcwd(), text=True)
    except Exception as exc:  # noqa: BLE001
        return f"[error] could not start background job: {exc}"
    _BG_SEQ += 1
    jid = f"bg{_BG_SEQ}"
    _BG_JOBS[jid] = {"proc": proc, "out_path": out.name, "command": command,
                     "started": time.time()}
    out.close()
    return (f"started background job {jid} (pid {proc.pid}): {command}\n"
            f"Use bash_output('{jid}') to read its output, bash_kill('{jid}') to stop it.")


def tool_bash_output(id, **_ignored):
    """Read accumulated output (and status) of a background job."""
    job = _BG_JOBS.get(id)
    if not job:
        return f"[error] no background job {id!r} (active: {', '.join(_BG_JOBS) or 'none'})"
    rc = job["proc"].poll()
    try:
        with open(job["out_path"], "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except OSError:
        data = ""
    if MAX_TOOL_OUTPUT and len(data) > MAX_TOOL_OUTPUT:
        data = data[-MAX_TOOL_OUTPUT:]
    status = "running" if rc is None else f"exited ({rc})"
    return f"job {id} [{status}] {job['command']}\n{data or '(no output yet)'}"


def tool_bash_kill(id, **_ignored):
    """Terminate a background job."""
    job = _BG_JOBS.get(id)
    if not job:
        return f"[error] no background job {id!r}"
    try:
        job["proc"].terminate()
    except Exception as exc:  # noqa: BLE001
        return f"[error] could not kill {id}: {exc}"
    return f"killed background job {id}"


TOOLS["bash_output"] = tool_bash_output
TOOLS["bash_kill"] = tool_bash_kill
SIDE_EFFECTS.add("bash_kill")
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "bash_output",
    "description": "Read the current output and status of a background job started with run_bash(run_in_background=true).",
    "parameters": {"type": "object", "properties": {
        "id": {"type": "string", "description": "The job id, e.g. 'bg1'"}}, "required": ["id"]},
}})
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "bash_kill",
    "description": "Stop a running background job by id.",
    "parameters": {"type": "object", "properties": {
        "id": {"type": "string", "description": "The job id, e.g. 'bg1'"}}, "required": ["id"]},
}})


# ----- user hooks -------------------------------------------------------------
def _inject_context(content, ctx):
    """Append hook-provided context to a user message (string or multimodal list)."""
    block = f"\n\n<context source=\"hook\">\n{ctx}\n</context>"
    if isinstance(content, list):
        return content + [{"type": "text", "text": block}]
    return (content or "") + block


_BLOCKING_HOOKS = ("PreToolUse", "UserPromptSubmit")
_CONTEXT_HOOKS = ("UserPromptSubmit", "SessionStart")


def _run_hooks(event, name=None, args=None, output=None, prompt=None):
    """Run configured hooks for an event.

    Returns a block-reason string if a blocking hook (PreToolUse /
    UserPromptSubmit) exits non-zero, else None. For UserPromptSubmit and
    SessionStart, any hook stdout is stashed in the module global `_HOOK_CONTEXT`
    so the caller can inject it as extra context. Hook failures never crash a turn.
    """
    global _HOOK_CONTEXT
    ctx = []
    for spec in (HOOKS.get(event) or []):
        if not isinstance(spec, dict) or not spec.get("command"):
            continue
        matcher = spec.get("matcher")
        if matcher and name is not None and not fnmatch.fnmatch(name, matcher):
            continue
        env = dict(os.environ, HERA_HOOK_EVENT=event)
        if name is not None:
            env["HERA_TOOL_NAME"] = name
        if args is not None:
            env["HERA_TOOL_ARGS"] = json.dumps(args)[:8000]
        if prompt is not None:
            env["HERA_USER_PROMPT"] = prompt[:8000]
        payload = json.dumps({"event": event, "tool": name, "args": args,
                              "prompt": prompt, "output": (output or "")[:4000]})
        try:
            r = subprocess.run(spec["command"], shell=True, input=payload, text=True,
                               capture_output=True, timeout=30, env=env, cwd=os.getcwd())
        except Exception:  # noqa: BLE001 SYM_EMDASH a broken hook must not break the agent
            continue
        if event in _BLOCKING_HOOKS and r.returncode != 0:
            reason = (r.stdout or r.stderr or "").strip() or f"blocked by {event} hook"
            return f"hook blocked: {reason[:200]}"
        if event in _CONTEXT_HOOKS and (r.stdout or "").strip():
            ctx.append(r.stdout.strip())
    if event in _CONTEXT_HOOKS:
        _HOOK_CONTEXT = "\n".join(ctx)
    return None


# ----- fine-grained permissions ----------------------------------------------
def _perm_match(entry, name, args):
    """Does a permission entry ('tool' or 'tool(<glob>)') match this call?"""
    m = re.match(r"^([A-Za-z_]+)(?:\((.*)\))?$", str(entry).strip())
    if not m or m.group(1) != name:
        return False
    glob = m.group(2)
    if glob is None:
        return True
    target = " ".join(str(args.get("command") or args.get("path") or "").split())
    return fnmatch.fnmatch(target, glob.strip())


def _perm_decision(name, args):
    """Return 'allow' / 'deny' / 'ask' from permissions, or None if no rule.

    Enterprise managed-policy rules are consulted FIRST and cannot be loosened by
    the user; user config is checked only if no managed rule matched."""
    for perms in (_MANAGED_PERMS, _PERMS):
        for bucket in ("deny", "ask", "allow"):
            for entry in (perms.get(bucket) or []):
                if _perm_match(entry, name, args):
                    return bucket
    return None


# ----- custom slash commands & named sub-agents ------------------------------
def _load_markdown_dir(path):
    """Return {name: text} for *.md files in `path` (name = filename without .md)."""
    out = {}
    try:
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".md"):
                try:
                    with open(os.path.join(path, fn), encoding="utf-8", errors="replace") as f:
                        out[fn[:-3]] = f.read()
                except OSError:
                    pass
    except OSError:
        pass
    return out


def _parse_frontmatter(text):
    """Split optional YAML-ish '---' frontmatter (key: value) from the body."""
    meta = {}
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4:].lstrip("\n")
    return meta, text


CUSTOM_COMMANDS = _load_markdown_dir(COMMANDS_DIR)
CUSTOM_AGENTS = _load_markdown_dir(AGENTS_DIR)


# ── Plugins & marketplaces (local, directory-based) ───────────────────────────
# A plugin is a directory under ~/.config/hera/plugins/<name>/ that may bundle:
#   plugin.json  (name/version/description/enabled)
#   commands/*.md   agents/*.md   mcp.json
# A marketplace is ~/.config/hera/marketplaces/<name>.json listing installable
# plugins ({name, description, source}); source is a local path or a git URL.
PLUGINS_DIR = os.path.join(CONFIG_DIR, "plugins")
MARKETPLACES_DIR = os.path.join(CONFIG_DIR, "marketplaces")
PLUGINS = []  # populated by load_plugins()


def load_plugins():
    """Discover enabled plugins, merge their commands/agents into the registries,
    and return a list of plugin info dicts (with any mcp.json path)."""
    found = []
    if not os.path.isdir(PLUGINS_DIR):
        return found
    for name in sorted(os.listdir(PLUGINS_DIR)):
        pdir = os.path.join(PLUGINS_DIR, name)
        if not os.path.isdir(pdir):
            continue
        manifest = {}
        mpath = os.path.join(pdir, "plugin.json")
        if os.path.isfile(mpath):
            try:
                with open(mpath, encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                manifest = {}
        if manifest.get("enabled") is False:
            continue
        cmds = _load_markdown_dir(os.path.join(pdir, "commands"))
        agents = _load_markdown_dir(os.path.join(pdir, "agents"))
        for k, v in cmds.items():
            CUSTOM_COMMANDS.setdefault(k, v)        # user's own dir wins on clash
        for k, v in agents.items():
            CUSTOM_AGENTS.setdefault(k, v)
        mcp_path = os.path.join(pdir, "mcp.json")
        found.append({"name": manifest.get("name", name), "dir": pdir,
                      "version": manifest.get("version", ""),
                      "description": manifest.get("description", ""),
                      "commands": list(cmds), "agents": list(agents),
                      "mcp": mcp_path if os.path.isfile(mcp_path) else None})
    PLUGINS[:] = found
    return found


def marketplace_catalog():
    """Aggregate installable plugins from all marketplace index files."""
    out = []
    if not os.path.isdir(MARKETPLACES_DIR):
        return out
    for fn in sorted(os.listdir(MARKETPLACES_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(MARKETPLACES_DIR, fn), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for p in (data.get("plugins") or []):
            if isinstance(p, dict) and p.get("name"):
                out.append({**p, "marketplace": fn[:-5]})
    return out


def install_plugin(name, source=None):
    """Install a plugin into PLUGINS_DIR from a local path or git URL. If `source`
    is omitted, resolve `name` against the marketplace catalog. Returns a message."""
    if source is None:
        match = next((p for p in marketplace_catalog() if p["name"] == name), None)
        if not match:
            return f"[error] no plugin {name!r} in any marketplace"
        source = match.get("source", "")
    dest = os.path.join(PLUGINS_DIR, name)
    if os.path.exists(dest):
        return f"[error] already installed: {dest}"
    os.makedirs(PLUGINS_DIR, exist_ok=True)
    src = os.path.expanduser(source)
    if os.path.isdir(src):                      # local path SYM_ARROW_R copy
        try:
            shutil.copytree(src, dest)
        except OSError as exc:
            return f"[error] copy failed: {exc}"
        return f"installed {name} {SYM_ARROW_L} {src}"
    if source.endswith(".git") or source.startswith(("http://", "https://", "git@")):
        proc = subprocess.run(["git", "clone", "--depth", "1", source, dest],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return f"[error] git clone failed: {(proc.stderr or '').strip()[:200]}"
        return f"installed {name} {SYM_ARROW_L} {source}"
    return f"[error] unusable source: {source!r}"


# ── Session persistence / resume ──────────────────────────────────────────────
def _compute_user_id():
    """Stable per-user id so sessions never mix on a shared machine.

    Identity comes from the API key: resolve_identity() asks the proxy who the
    key belongs to and caches the account email as `user`, so the key alone
    labels your sessions. HERA_USER still overrides; before any email is known
    we fall back to a hash of the key (each Open WebUI user has their own key)."""
    u = _cfg("HERA_USER", key="user")
    if u:
        return re.sub(r"[^A-Za-z0-9._@-]", "_", u)[:64]
    if API_KEY:
        return "key-" + hashlib.sha256(API_KEY.encode()).hexdigest()[:12]
    return "default"


def _sessions_dir_for(uid):
    return os.path.join(
        os.path.expanduser(_env("HERA_SESSIONS_DIR", default="~/.config/hera/sessions")),
        uid)


USER_ID = _compute_user_id()
SESSIONS_DIR = _sessions_dir_for(USER_ID)
# Friendly identity (resolved from the key via the proxy's /whoami): the account
# email and display name, so every surface can greet the real user.
USER_EMAIL = _cfg("HERA_USER", key="user")
USER_NAME = _FILE_CFG.get("user_name", "")


def whoami_label():
    """A human label for the signed-in user: 'Name (email)' / email / (not set)."""
    if USER_NAME and USER_EMAIL:
        return f"{USER_NAME} ({USER_EMAIL})"
    return USER_EMAIL or USER_NAME or "(not signed in)"


def _whoami_url():
    """The proxy's identity endpoint, derived from the API URL (strip /v1)."""
    base = API_URL[:-3] if API_URL.endswith("/v1") else API_URL
    return base.rstrip("/") + "/whoami"


def _skills_url():
    """The proxy's shared-skills catalog endpoint, derived from the API URL."""
    base = API_URL[:-3] if API_URL.endswith("/v1") else API_URL
    return base.rstrip("/") + "/skills"


def fetch_shared_skills():
    if not (API_URL and API_KEY):
        return None, "no server or API key configured"
    try:
        r = requests.get(
            _skills_url(),
            timeout=4,
            headers={"Authorization": f"Bearer {API_KEY}", "User-Agent": _WEB_UA},
        )
    except requests.exceptions.RequestException as exc:
        return None, str(exc)
    if not r.ok:
        return None, f"{r.status_code}: {(r.text or '').strip()[:200]}"
    try:
        data = r.json()
    except ValueError as exc:
        return None, f"invalid JSON from proxy: {exc}"
    return data.get("skills") or [], ""


def print_shared_skills(skill_id=""):
    skills, err = fetch_shared_skills()
    if skills is None:
        print(f"\n{DIM}shared skills unavailable: {err}{R}\n")
        return
    if not skills:
        print(f"\n{DIM}no shared skills are configured on the server.{R}\n")
        return
    if skill_id:
        target = next((s for s in skills if s.get("id") == skill_id), None)
        if not target:
            print(f"\n{DIM}no shared skill named {skill_id!r}.{R}\n")
            return
        state = "enabled" if target.get("enabled") else "disabled"
        triggers = ", ".join(target.get("triggers") or []) or "(none)"
        print(
            f"\n{CYAN}{target.get('id')}{R}  {DIM}[{state}] {target.get('provider') or 'prompt'}"
            f"  priority={target.get('priority', 0)}{R}\n"
            f"  {target.get('description') or '(no description)'}\n"
            f"  triggers: {triggers}\n"
        )
        return
    print(f"\n{DIM}shared skills:{R}")
    for skill in skills:
        state = "enabled" if skill.get("enabled") else "disabled"
        provider = skill.get("provider") or "prompt"
        print(
            f"  {CYAN}{skill.get('id'):<18}{R}{DIM}[{state:<8}] {provider:<10} "
            f"prio={skill.get('priority', 0):<3}{R} {skill.get('description', '')}"
        )
    print(f"\n{DIM}use /skills <id> for detail. Explicit activation: @skill:<id> or /skill <id>{R}\n")


def resolve_identity(force=False):
    """Make the API key *be* the identity: ask the proxy who this key belongs to
    and cache the account email, so sessions are labelled by the real user with
    no HERA_USER to set. Idempotent and fail-silent SYM_EMDASH falls back to the key hash
    if the proxy is old or unreachable. Returns the known/resolved email.

    Pass force=True (e.g. `hera whoami`) to bypass the cache and always query
    the API so the displayed identity reflects the current key."""
    global USER_ID, SESSIONS_DIR, USER_EMAIL, USER_NAME
    known = _env("HERA_USER") or _FILE_CFG.get("user")
    if known and not force:
        USER_EMAIL = known
        USER_NAME = USER_NAME or _FILE_CFG.get("user_name", "")
        return known
    if not (API_URL and API_KEY):
        return ""
    try:
        r = requests.get(_whoami_url(), timeout=4,
                         headers={"Authorization": f"Bearer {API_KEY}",
                                  "User-Agent": _WEB_UA})
        if r.ok:
            data = r.json() or {}
            email = (data.get("email") or "").strip()
            name = (data.get("name") or "").strip()
            if email:
                USER_EMAIL = email
                USER_NAME = name
                save_config({"user": email, "user_name": name})
                USER_ID = _compute_user_id()
                SESSIONS_DIR = _sessions_dir_for(USER_ID)
                if sys.stdin.isatty():
                    greet = f"{name} ({email})" if name else email
                    print(f"{GREEN}{SYM_CHECK} signed in as {greet}{R} "
                          f"{DIM}{SYM_EMDASH} sessions are labelled by your account.{R}")
                return email
    except (requests.exceptions.RequestException, ValueError):
        pass
    return ""
CURRENT_SESSION = {"id": None, "created": None, "title": None}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_session_id():
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def save_session(messages):
    """Persist the conversation (skips trivial sessions). Best-effort."""
    if not CURRENT_SESSION["id"] or len(messages) <= 1:
        return
    # Capture the first real user question once, as a human title (survives
    # compaction, which would later replace the messages with a summary).
    if not CURRENT_SESSION.get("title"):
        t = _first_user(messages)
        if t and t != "(empty)":
            CURRENT_SESSION["title"] = t
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        state = {
            "id": CURRENT_SESSION["id"],
            "created": CURRENT_SESSION["created"],
            "updated": _now(),
            "cwd": os.getcwd(),
            "model": MODEL,
            "title": CURRENT_SESSION.get("title"),
            "tokens": dict(SESSION),
            "messages": messages,
        }
        path = os.path.join(SESSIONS_DIR, CURRENT_SESSION["id"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass


def list_sessions():
    out = []
    if not os.path.isdir(SESSIONS_DIR):
        return out
    for fn in os.listdir(SESSIONS_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fn), encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    out.sort(key=lambda s: s.get("updated", ""), reverse=True)
    return out


def _same_project(s):
    """True if saved session `s` was started in the current working directory."""
    try:
        return os.path.realpath(s.get("cwd") or "") == os.path.realpath(os.getcwd())
    except OSError:
        return False


def project_sessions():
    """Saved sessions that belong to the current project (cwd), newest first."""
    return [s for s in list_sessions() if _same_project(s)]


def load_session(sid):
    """Resolve `sid` (exact id, prefix, or '__latest__') to a saved session."""
    sessions = list_sessions()
    if not sessions:
        return None
    chosen = None
    if sid in ("__latest__", "", None):
        # `hera --continue` resumes the latest conversation *in this project*,
        # falling back to the global latest if this project has none yet.
        here = project_sessions()
        chosen = here[0] if here else sessions[0]
    else:
        for s in sessions:
            if s.get("id") == sid:
                chosen = s
                break
        if chosen is None:
            for s in sessions:
                if s.get("id", "").startswith(sid):
                    chosen = s
                    break
    if chosen is not None:
        # Heal any malformed tool_calls saved before the normalization fix,
        # so resuming an old session can't 500 on the first request.
        _sanitize_history(chosen.get("messages") or [])
    return chosen


def _first_user(messages):
    for m in messages:
        if m.get("role") != "user":
            continue
        text = " ".join(_text_of(m.get("content")).split())
        if text.startswith("[Summary of earlier conversation]"):
            continue  # a compacted session SYM_EMDASH keep looking for the real question
        if text:
            return text[:64]
    return "(empty)"


def _session_label(s):
    """Human title for a saved session: its stored first question (preferred), or
    the first user message in its history."""
    return s.get("title") or _first_user(s.get("messages") or [])


def print_sessions(project_only=False):
    """List saved conversations. project_only SYM_ARROW_R just this project's (for the
    interactive `/sessions`); otherwise all of them (for `hera --list-sessions`)."""
    alls = list_sessions()
    sessions = [s for s in alls if _same_project(s)] if project_only else alls
    if not sessions:
        if project_only and alls:
            print(f"\n{DIM}no conversations in this project yet "
                  f"({len(alls)} in other projects {SYM_EMDASH} `hera --list-sessions` to see all).{R}\n")
        else:
            print(f"\n{DIM}no saved conversations yet.{R}\n")
        return
    scope = "in this project" if project_only else "all projects"
    print(f"\n{BOLD}Saved conversations{R} {DIM}({scope}, newest first) {SYM_EMDASH} `/resume` to pick one, "
          f"or `hera --continue` for the latest{R}")
    for i, s in enumerate(sessions[:20], 1):
        msgs = s.get("messages", [])
        nturns = sum(1 for m in msgs if m.get("role") == "user")
        proj = os.path.basename((s.get("cwd") or "").rstrip("/")) or "~"
        print(f"  {ACCENT}{i:>2}.{R} {_session_label(s)}")
        print(f"      {DIM}{(s.get('updated','') or '')[:16]} {SYM_MIDDOT} {nturns} message(s) {SYM_MIDDOT} {proj}/{R}")
    other = len(alls) - len(sessions)
    if project_only and other > 0:
        print(f"  {DIM}{SYM_ELLIPSIS} {other} more in other projects {SYM_EMDASH} `hera --list-sessions` to see all{R}")
    print()


def _switch_to(messages, s):
    """Load saved session `s` into the live `messages`, replacing the current one."""
    save_session(messages)  # keep the current conversation before switching
    messages[:] = s["messages"]
    CURRENT_SESSION["id"] = s["id"]
    CURRENT_SESSION["created"] = s.get("created")
    CURRENT_SESSION["title"] = s.get("title") or _first_user(s.get("messages") or [])
    SESSION.update(s.get("tokens", {}))
    _always_ok.clear()
    reset_turn_marks()  # marks belong to the previous conversation
    nturns = sum(1 for m in messages if m.get("role") == "user")
    where = s.get("cwd", "")
    print(f"\n{GREEN}resumed:{R} {_session_label(s)} {DIM}({nturns} message(s), "
          f"{SESSION.get('total', 0)} tok){R}")
    if where and where != os.getcwd():
        print(f"{DIM}note: this session was started in {_short(where)} {SYM_EMDASH} "
              f"you're now in {_short(os.getcwd())}{R}")
    print()


def resume_picker(messages):
    """Show this project's recent conversations and let the user pick one."""
    all_sessions = list_sessions()
    sessions = [s for s in all_sessions if _same_project(s)]
    if not sessions:
        other = len(all_sessions)
        if other:
            print(f"\n{DIM}no conversations in this project yet "
                  f"({other} in other projects {SYM_EMDASH} `hera --list-sessions` to see all).{R}\n")
        else:
            print(f"\n{DIM}no saved conversations yet.{R}\n")
        return
    shown = sessions[:20]
    print(f"\n{BOLD}Resume a conversation{R} {DIM}(this project, newest first){R}")
    for i, s in enumerate(shown, 1):
        msgs = s.get("messages", [])
        nturns = sum(1 for m in msgs if m.get("role") == "user")
        print(f"  {ACCENT}{i:>2}.{R} {_session_label(s)}")
        print(f"      {DIM}{(s.get('updated','') or '')[:16]} {SYM_MIDDOT} {nturns} message(s){R}")
    try:
        ans = input(f"\n{BOLD}  number to resume (Enter to cancel):{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not ans:
        print(f"{DIM}cancelled.{R}\n")
        return
    if not ans.isdigit() or not (1 <= int(ans) <= len(shown)):
        print(f"{DIM}'{ans}' is not one of 1{SYM_ENDASH}{len(shown)}.{R}\n")
        return
    _switch_to(messages, shown[int(ans) - 1])


# ── Extensions: MCP servers + custom tools ────────────────────────────────────
MCP_CONFIG = os.path.expanduser(_env("HERA_MCP_CONFIG",
                                     default="~/.config/hera/mcp.json"))
MCP_SANDBOX = _truthy(_env("HERA_MCP_SANDBOX"))  # run MCP servers under the sandbox
CUSTOM_TOOLS_PATHS = [
    os.path.expanduser("~/.config/hera/tools.py"),
    os.path.join(os.getcwd(), ".hera", "tools.py"),
]
EXT_TOOLS = set()      # tool names added by extensions (require approval)
_mcp_clients = []


def _safe_tool_name(name):
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name[:64]


class McpClient:
    """Minimal MCP stdio client (newline-delimited JSON-RPC 2.0)."""

    def __init__(self, name, command, args=None, env=None):
        self.name = name
        full_env = dict(os.environ)
        full_env.update(env or {})
        argv = [command] + list(args or [])
        if MCP_SANDBOX:
            argv = _sandbox_wrap_argv(argv)
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=full_env,
        )
        self._id = 0
        self.tools = []
        self._handshake()

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method, params=None, timeout=30):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"MCP '{self.name}' timed out on {method}")
            ready, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError(f"MCP '{self.name}' timed out on {method}")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP '{self.name}' closed the connection")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip non-JSON log noise on stdout
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"].get("message", "MCP error"))
                return msg.get("result", {})
            # otherwise a notification / unrelated message → ignore

    def _notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _handshake(self):
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hera", "version": "1"},
        })
        self._notify("notifications/initialized")
        self.tools = self._rpc("tools/list").get("tools", [])

    def call(self, tool, arguments):
        res = self._rpc("tools/call", {"name": tool, "arguments": arguments or {}})
        parts = []
        for c in res.get("content", []):
            parts.append(c.get("text", "") if c.get("type") == "text" else json.dumps(c))
        text = "\n".join(p for p in parts if p) or "(no output)"
        return ("[mcp error] " + text) if res.get("isError") else text

    def close(self):
        try:
            self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def _expand_env_vars(s):
    """Expand ${VAR} from the environment so secrets stay out of mcp.json."""
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                  lambda m: os.environ.get(m.group(1), ""), str(s))


def _mcp_headers(spec):
    """Build request headers for an HTTP MCP server, including bearer auth.

    `token`/`auth_token` becomes `Authorization: Bearer SYM_ELLIPSIS` SYM_EMDASH this is the
    credential an OAuth flow ultimately yields (a personal access / API token).
    `headers` lets you set anything else. Both support ${ENV} expansion.
    """
    headers = {k: _expand_env_vars(v) for k, v in (spec.get("headers") or {}).items()}
    token = spec.get("token") or spec.get("auth_token")
    if token and "Authorization" not in headers:
        headers["Authorization"] = "Bearer " + _expand_env_vars(token)
    return headers


class McpHttpClient:
    """MCP client over Streamable HTTP SYM_EMDASH JSON-RPC POSTed to one endpoint, with
    either an application/json reply or a text/event-stream (SSE) reply. Works
    with remote MCP servers that authenticate via a bearer/OAuth token."""

    def __init__(self, name, url, headers=None, timeout=30):
        self.name = name
        self.url = url
        self.timeout = timeout
        self.session_id = None
        self._id = 0
        self.tools = []
        self.headers = {"Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"}
        self.headers.update(headers or {})
        self._handshake()

    def _post(self, payload, timeout=None):
        h = dict(self.headers)
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        resp = requests.post(self.url, json=payload, headers=h,
                             timeout=timeout or self.timeout, stream=True)
        resp.raise_for_status()
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        if "text/event-stream" in resp.headers.get("Content-Type", ""):
            for raw in resp.iter_lines(decode_unicode=True):
                if raw and raw.startswith("data:"):
                    data = raw[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                        return msg
            return None
        text = (resp.text or "").strip()
        return json.loads(text) if text else None

    def _rpc(self, method, params=None, timeout=None):
        self._id += 1
        msg = self._post({"jsonrpc": "2.0", "id": self._id, "method": method,
                          "params": params or {}}, timeout)
        if msg is None:
            raise RuntimeError(f"MCP '{self.name}' returned no response to {method}")
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message", "MCP error"))
        return msg.get("result", {})

    def _notify(self, method, params=None):
        try:
            self._post({"jsonrpc": "2.0", "method": method, "params": params or {}})
        except Exception:  # noqa: BLE001
            pass

    def _handshake(self):
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "hera", "version": "1"}})
        self._notify("notifications/initialized")
        self.tools = self._rpc("tools/list").get("tools", [])

    def call(self, tool, arguments):
        res = self._rpc("tools/call", {"name": tool, "arguments": arguments or {}})
        parts = []
        for c in res.get("content", []):
            parts.append(c.get("text", "") if c.get("type") == "text" else json.dumps(c))
        text = "\n".join(p for p in parts if p) or "(no output)"
        return ("[mcp error] " + text) if res.get("isError") else text

    def close(self):
        if not self.session_id:
            return
        try:  # politely end the server session if it tracks one
            requests.delete(self.url, headers={"Mcp-Session-Id": self.session_id,
                            **{k: v for k, v in self.headers.items() if k == "Authorization"}},
                            timeout=5)
        except Exception:  # noqa: BLE001
            pass


def _register_tool(name, description, parameters, func, read_only=False):
    TOOLS[name] = func
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": name, "description": description or name,
        "parameters": parameters or {"type": "object", "properties": {}},
    }})
    EXT_TOOLS.add(name)
    if not read_only:
        SIDE_EFFECTS.add(name)


def _register_mcp_file(path):
    """Start the MCP servers declared in one config file; return tool names."""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{YELL}[mcp] bad config {path}: {exc}{R}", file=sys.stderr)
        return []
    servers = cfg.get("mcpServers", cfg if isinstance(cfg, dict) else {})
    loaded = []
    for sname, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        try:
            if spec.get("url"):                       # remote HTTP/SSE MCP server
                client = McpHttpClient(sname, _expand_env_vars(spec["url"]),
                                       _mcp_headers(spec), int(spec.get("timeout", 30)))
            elif spec.get("command"):                 # local stdio MCP server
                client = McpClient(sname, spec["command"], spec.get("args"), spec.get("env"))
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"{YELL}[mcp] failed to start '{sname}': {exc}{R}", file=sys.stderr)
            continue
        _mcp_clients.append(client)
        for t in client.tools:
            full = _safe_tool_name(f"mcp__{sname}__{t['name']}")
            run = (lambda cl, tn: (lambda **kw: cl.call(tn, kw)))(client, t["name"])
            _register_tool(full, t.get("description", f"MCP tool {t['name']} ({sname})"),
                           t.get("inputSchema"), run)
            loaded.append(full)
    return loaded


def register_mcp():
    """Load the main mcp.json plus any mcp.json bundled by installed plugins."""
    loaded = []
    if os.path.isfile(MCP_CONFIG):
        loaded += _register_mcp_file(MCP_CONFIG)
    for p in PLUGINS:
        if p.get("mcp"):
            loaded += _register_mcp_file(p["mcp"])
    return loaded


# ── MCP OAuth (authorization-code + PKCE) ─────────────────────────────────────
def _pkce_pair():
    """Return (verifier, S256 challenge) for OAuth 2.0 PKCE (RFC 7636)."""
    import secrets
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _oauth_authorize_url(auth_endpoint, client_id, redirect_uri, challenge, scope="", state=""):
    params = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
              "code_challenge": challenge, "code_challenge_method": "S256"}
    if scope:
        params["scope"] = scope
    if state:
        params["state"] = state
    sep = "&" if "?" in auth_endpoint else "?"
    return auth_endpoint + sep + _urlparse.urlencode(params)


def _discover_oauth_endpoints(server_url):
    """Find a server's OAuth endpoints via RFC 8414 / OIDC well-known metadata."""
    parts = _urlparse.urlsplit(server_url)
    base = f"{parts.scheme}://{parts.netloc}"
    for path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        try:
            r = requests.get(base + path, timeout=10)
            r.raise_for_status()
            d = r.json()
            if d.get("authorization_endpoint") and d.get("token_endpoint"):
                return {"authorization_endpoint": d["authorization_endpoint"],
                        "token_endpoint": d["token_endpoint"],
                        "scopes_supported": d.get("scopes_supported", [])}
        except Exception:  # noqa: BLE001 SYM_EMDASH try the next well-known path
            continue
    return {}


def _save_mcp_token(name, token):
    """Persist an obtained OAuth token onto the MCP server entry in mcp.json."""
    if not os.path.isfile(MCP_CONFIG):
        return False
    try:
        with open(MCP_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    servers = cfg.get("mcpServers", cfg if isinstance(cfg, dict) else {})
    if name not in servers or not isinstance(servers[name], dict):
        return False
    servers[name]["token"] = token
    try:
        with open(MCP_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except OSError:
        return False


def mcp_oauth_login(name):
    """Interactive OAuth 2.0 (auth-code + PKCE) for an HTTP MCP server, saving the
    token to mcp.json. Endpoints come from the server's `oauth` block or RFC 8414
    discovery. NOTE: the live browser + token-exchange flow is best-effort and is
    not exercised by the test suite (no OAuth server available here)."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    if not name:
        return "[error] usage: /mcp login <server>"
    if not os.path.isfile(MCP_CONFIG):
        return f"[error] no {MCP_CONFIG}"
    try:
        with open(MCP_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return f"[error] bad mcp.json: {exc}"
    servers = cfg.get("mcpServers", cfg if isinstance(cfg, dict) else {})
    spec = servers.get(name)
    if not isinstance(spec, dict) or not spec.get("url"):
        return f"[error] no HTTP MCP server named {name!r}"
    oauth = spec.get("oauth") or {}
    eps = {"authorization_endpoint": oauth.get("authorization_endpoint"),
           "token_endpoint": oauth.get("token_endpoint")}
    if not (eps["authorization_endpoint"] and eps["token_endpoint"]):
        eps = _discover_oauth_endpoints(spec["url"]) or eps
    if not (eps.get("authorization_endpoint") and eps.get("token_endpoint")):
        return ("[error] couldn't find OAuth endpoints SYM_EMDASH set oauth.authorization_endpoint "
                "and oauth.token_endpoint in mcp.json")
    client_id = oauth.get("client_id") or "hera"
    scope = oauth.get("scope") or " ".join(eps.get("scopes_supported") or [])
    verifier, challenge = _pkce_pair()
    holder = {}

    class _CB(BaseHTTPRequestHandler):
        def do_GET(self):
            q = _urlparse.parse_qs(_urlparse.urlsplit(self.path).query)
            holder["code"] = (q.get("code") or [None])[0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Hera: authorization received. You can close this tab.")

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), _CB)
    redirect = f"http://127.0.0.1:{httpd.server_address[1]}/callback"
    url = _oauth_authorize_url(eps["authorization_endpoint"], client_id, redirect, challenge, scope)
    print(f"\n  Opening your browser to authorize '{name}'{SYM_ELLIPSIS}\n  {url}\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    threading.Thread(target=httpd.handle_request, daemon=True).start()
    for _ in range(240):  # ~120s
        if holder.get("code"):
            break
        time.sleep(0.5)
    httpd.server_close()
    code = holder.get("code")
    if not code:
        return "[error] no authorization code received (timed out)"
    try:
        r = requests.post(eps["token_endpoint"], data={
            "grant_type": "authorization_code", "code": code, "redirect_uri": redirect,
            "client_id": client_id, "code_verifier": verifier}, timeout=30)
        r.raise_for_status()
        token = r.json().get("access_token")
    except Exception as exc:  # noqa: BLE001
        return f"[error] token exchange failed: {exc}"
    if not token:
        return "[error] no access_token in the response"
    _save_mcp_token(name, token)
    return f"authorized '{name}' {SYM_EMDASH} token saved to {MCP_CONFIG} (reload to use it)"


def register_custom_tools():
    loaded = []
    for i, path in enumerate(CUSTOM_TOOLS_PATHS):
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"hera_user_tools_{i}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001
            print(f"{YELL}[tools] failed to load {path}: {exc}{R}", file=sys.stderr)
            continue
        for t in getattr(mod, "HERA_TOOLS", []):
            try:
                _register_tool(_safe_tool_name(t["name"]), t.get("description", ""),
                               t.get("parameters"), t["run"], t.get("read_only", False))
                loaded.append(t["name"])
            except (KeyError, TypeError):
                pass
    return loaded


def register_semantic_search():
    """Register the embeddings-backed semantic_search tool iff an endpoint works."""
    if not embeddings_available():
        return False
    TOOLS["semantic_search"] = tool_semantic_search
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": "semantic_search",
        "description": ("Rank code chunks by embedding similarity to a natural-language "
                        "query (returns path:line: (score) snippet). Use for fuzzy "
                        "'where is the logic that SYM_ELLIPSIS' questions."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural-language query"},
            "path":  {"type": "string", "description": "Directory to search (default '.')"},
            "k":     {"type": "integer", "description": "Number of results (default 8)"},
        }, "required": ["query"]},
    }})
    return True


def register_extensions(quiet=False):
    plugins = load_plugins()   # merges plugin commands/agents before the rest load
    mcp = register_mcp()
    custom = register_custom_tools()
    out = sys.stderr if quiet else sys.stdout   # serve mode keeps stdout JSON-only
    if plugins:
        print(f"{DIM}[ext] loaded {len(plugins)} plugin(s): "
              f"{', '.join(p['name'] for p in plugins[:6])}"
              f"{'{SYM_ELLIPSIS}' if len(plugins) > 6 else ''}{R}", file=out)
    if mcp:
        print(f"{DIM}[ext] loaded {len(mcp)} MCP tool(s): {', '.join(mcp[:6])}"
              f"{'{SYM_ELLIPSIS}' if len(mcp) > 6 else ''}{R}", file=out)
    if custom:
        print(f"{DIM}[ext] loaded {len(custom)} custom tool(s): {', '.join(custom)}{R}", file=out)
    if register_semantic_search():
        print(f"{DIM}[ext] semantic_search enabled (embeddings at {EMBED_URL}){R}", file=out)


def close_extensions():
    for c in _mcp_clients:
        c.close()


# ── Memory / project context (Claude-Code-style hierarchy) ────────────────────
CONTEXT_FILES = ("HERA.md", "AGENTS.md", "AGENT.md")
MEMORY_MAX_PER_FILE = 16000   # cap one source so a giant AGENT.md can't dominate
MEMORY_MAX_TOTAL    = 32000   # overall cap across all memory sources


def _memory_sources():
    """Ordered (scope, path) memory files, least- to most-specific:
    enterprise SYM_ARROW_R user (~/.config/hera) SYM_ARROW_R project tree (filesystem root down to
    cwd). Mirrors Claude Code's CLAUDE.md hierarchy."""
    out = []
    ent = _env("HERA_ENTERPRISE_MEMORY") or "/etc/hera/HERA.md"
    if os.path.isfile(ent):
        out.append(("enterprise", ent))
    user = os.path.join(CONFIG_DIR, "HERA.md")
    if os.path.isfile(user):
        out.append(("user", user))
    # Walk up from cwd collecting the first context file at each level, then add
    # them root-first so the most-specific (cwd) lands last and wins.
    chain, d = [], os.getcwd()
    while True:
        for fn in CONTEXT_FILES:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                chain.append(("project", p))
                break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    out.extend(reversed(chain))
    return out


def _expand_memory_imports(text, base_dir, seen, depth=0):
    """Inline `@path` import lines (recursive, cycle- and depth-guarded), like
    Claude Code's CLAUDE.md @imports. A line that is just `@some/file` is
    replaced with that file's (also expanded) contents."""
    if depth > 5:
        return text
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*@(\S+)\s*$", line)
        if m:
            p = os.path.expanduser(m.group(1))
            if not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            p = os.path.realpath(p)
            if os.path.isfile(p) and p not in seen:
                seen.add(p)
                try:
                    sub = open(p, encoding="utf-8", errors="replace").read()
                    out.append(f"<!-- imported: {m.group(1)} -->")
                    out.append(_expand_memory_imports(sub, os.path.dirname(p), seen, depth + 1))
                    continue
                except OSError:
                    pass
        out.append(line)
    return "\n".join(out)


def load_memory():
    """All memory sources as a list of (scope, path, expanded_body)."""
    seen, parts = set(), []
    for scope, path in _memory_sources():
        rp = os.path.realpath(path)
        if rp in seen:
            continue
        seen.add(rp)
        try:
            body = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        body = _expand_memory_imports(body, os.path.dirname(path), seen)[:MEMORY_MAX_PER_FILE]
        if body.strip():
            parts.append((scope, path, body))
    return parts


def add_memory(text, scope="project"):
    """Append a `- fact` line to project (./HERA.md) or user memory. Powers `#`."""
    text = text.strip()
    if not text:
        return "nothing to remember"
    path = (os.path.join(CONFIG_DIR, "HERA.md") if scope == "user"
            else os.path.join(os.getcwd(), "HERA.md"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fresh = not os.path.exists(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            if fresh:
                f.write(f"# {'User' if scope == 'user' else 'Project'} memory (Hera)\n\n")
            f.write(f"- {text}\n")
    except OSError as exc:
        return f"[error] could not write memory: {exc}"
    return f"remembered ({scope}): {path}"


def load_project_context():
    """Back-compat shim: (filename, combined_body) of all memory, or (None, None).
    Kept for the banner/startup code that wants a single label."""
    parts = load_memory()
    if not parts:
        return None, None
    label = ", ".join(sorted({os.path.basename(p) for _, p, _ in parts}))
    body = "\n\n".join(b for _, _, b in parts)[:MEMORY_MAX_TOTAL]
    return label, body


def memory_report():
    """`/memory`: show the loaded memory hierarchy and how to add to it."""
    parts = load_memory()
    if not parts:
        print(f"\n{DIM}no memory yet. Add a fact with {R}{CYAN}# <fact>{R}{DIM} (project) "
              f"or {R}{CYAN}# user <fact>{R}{DIM} (all projects).{R}\n")
        return
    print(f"\n{BOLD}Memory{R} {DIM}(loaded into every prompt; most-specific last){R}")
    for scope, path, body in parts:
        print(f"  {CYAN}{scope:10}{R}{DIM}{path}  ({len(body)} chars){R}")
    print(f"{DIM}  add: {R}{CYAN}# <fact>{R}{DIM} {SYM_MIDDOT} {R}{CYAN}# user <fact>{R}{DIM} {SYM_MIDDOT} "
          f"edit a file directly{R}\n")


def _init_prompt():
    """Instruction for `/init`: have the agent analyze the repo and write HERA.md."""
    cmds = _detect_project_commands()
    hint = (" Detected toolchain: " + ", ".join(cmds) + ".") if cmds else ""
    return ("Analyze this project and create a concise HERA.md at the repo root that future Hera "
            "sessions load as context. Use glob/search/read_file to inspect the structure, key "
            "entry points, build/test/run commands, conventions, and any gotchas. Keep it under "
            "~150 skimmable lines with short sections (Overview, Layout, Build/Test/Run, "
            "Conventions). Write it with write_file, then stop." + hint)


def export_conversation(messages, path=None):
    """`/export`: write the conversation to Markdown (default) or JSON (.json path)."""
    if not path:
        path = os.path.join(os.getcwd(), f"hera-conversation-{time.strftime('%Y%m%d-%H%M%S')}.md")
    path = _resolve(path)
    if path.endswith(".json"):
        data = [m for m in messages if m.get("role") != "system"]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            return f"[error] {exc}"
        return f"exported {len(data)} messages {SYM_ARROW_R} {path}"
    lines = [f"# Hera conversation {SYM_EMDASH} {time.strftime('%Y-%m-%d %H:%M')}"]
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        c = _text_of(m.get("content"))
        if role == "user":
            lines.append(f"\n## You\n\n{c}")
        elif role == "assistant":
            if c.strip():
                lines.append(f"\n## Hera\n\n{c}")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                lines.append(f"\n> 🔧 `{fn.get('name')}` {str(fn.get('arguments', ''))[:300]}")
        elif role == "tool":
            lines.append(f"\n> {SYM_INTERRUPT} {c[:500]}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as exc:
        return f"[error] {exc}"
    return f"exported {SYM_ARROW_R} {path}"


def add_permission(bucket, rule):
    """Add a live permission rule (deny/ask/allow) and persist it. Powers
    `/permissions <bucket> <rule>`."""
    if bucket not in ("deny", "ask", "allow"):
        return "[error] bucket must be deny, ask, or allow"
    rule = (rule or "").strip()
    if not rule:
        return "[error] empty rule (e.g. run_bash(rm *) or edit_file(src/**))"
    perms = dict(_PERMS) if isinstance(_PERMS, dict) else {}
    lst = list(perms.get(bucket) or [])
    if rule not in lst:
        lst.append(rule)
    perms[bucket] = lst
    globals()["_PERMS"] = perms
    save_config({"permissions": perms})
    return f"added {bucket} rule: {rule}  ({len(lst)} {bucket} total)"


def _config_summary():
    """Key effective settings (for `/config`)."""
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "api_url": API_URL or "(unset)",
        "auto_mode": AUTO_MODE,
        "plan_mode": PLAN_MODE,
        "think": THINK_LEVEL,
        "output_style": OUTPUT_STYLE,
        "web": WEB_ENABLED,
        "auto_verify": AUTO_VERIFY,
        "telemetry": "on" if TELEMETRY_ON else "off",
        "sandbox": sandbox_label(),
        "config_path": CONFIG_PATH,
    }


def _doctor_get(url, timeout=4):
    """Silent GET; returns (status_code, True) or (None, False)."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code, True
    except Exception:
        return None, False


def _derive_cli_server():
    """Derive the CLI file-server URL (port 8081) from API_URL."""
    if not API_URL:
        return None
    return re.sub(r":\d+(/.*)?$", ":8081", API_URL.rstrip("/"))


def _self_update(cli_base):
    """Download latest hera.py from cli_base and atomically replace this script."""
    hera_path = os.path.abspath(__file__)
    try:
        resp = requests.get(f"{cli_base}/hera.py", timeout=60, stream=True)
        resp.raise_for_status()
        tmp = hera_path + ".new"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        os.chmod(tmp, os.stat(hera_path).st_mode)
        os.replace(tmp, hera_path)
        return True, None
    except Exception as e:
        return False, str(e)


def _run_doctor():
    """All-in-one: self-update + Docker stack health + CLI diagnostics/auto-fix."""
    ok_s   = f"{GREEN}{SYM_CHECK}{R}"
    fail_s = f"{YELL}{SYM_CROSS}{R}"
    warn_s = f"{YELL}~{R}"
    fix_s  = f"{GREEN}{SYM_ARROW_U}{R}"

    print(f"\n{BOLD}Hera Doctor{R}\n")

    # ── 1. Self-update ──────────────────────────────────────────────────────
    print(f"  {BOLD}Version{R}")
    cli_base = _derive_cli_server()
    print(f"    current : {VERSION}")
    if cli_base:
        try:
            latest = requests.get(f"{cli_base}/VERSION", timeout=4).text.strip()
            if latest == VERSION:
                print(f"    {ok_s} up to date")
            else:
                print(f"    {warn_s} update available: {latest}")
                ans = input(f"    {CYAN}Update now? [y/N]{R} ").strip().lower()
                if ans == "y":
                    print("    downloadingSYM_ELLIPSIS", end="", flush=True)
                    ok, err = _self_update(cli_base)
                    print(f"\r    {fix_s if ok else fail_s} "
                          f"{'updated {SYM_EMDASH} restart hera to apply' if ok else f'failed: {err}'}")
        except Exception as e:
            print(f"    {warn_s} version check failed: {e}")
    else:
        print(f"    {warn_s} HERA_API_URL not set {SYM_EMDASH} cannot reach CLI server")

    # ── 2. Docker stack health ──────────────────────────────────────────────
    print(f"\n  {BOLD}Stack health{R}")
    if API_URL:
        m = re.search(r"https?://([^:/]+)", API_URL)
        host = m.group(1) if m else "localhost"
        m2 = re.search(r":(\d+)", API_URL)
        proxy_port = int(m2.group(1)) if m2 else 8090
        services = [
            ("auth-proxy",  host, proxy_port, "/health"),
            ("qwen-server", host, 8080,        "/health"),
            ("open-webui",  host, 3000,        "/health"),
            ("grafana",     host, 3001,        "/api/health"),
            ("prometheus",  host, 9090,        "/-/healthy"),
            ("gpustack",    host, 8888,        "/"),
            ("cli-server",  host, 8081,        "/VERSION"),
        ]
        for name, h, port, path in services:
            code, up = _doctor_get(f"http://{h}:{port}{path}")
            sym = ok_s if up else fail_s
            detail = f"HTTP {code}" if code else "unreachable"
            print(f"    {sym} {name:<14}: {detail}")
    else:
        print(f"    {warn_s} HERA_API_URL not set {SYM_EMDASH} cannot check stack")

    # ── 3. CLI diagnostics + auto-fix ──────────────────────────────────────
    print(f"\n  {BOLD}CLI diagnostics{R}")

    print(f"    {ok_s if API_KEY else fail_s} API key     : "
          f"{'set' if API_KEY else 'NOT SET {SYM_EMDASH} run hera to configure'}")
    print(f"    {ok_s} model       : {MODEL}")
    print(f"    {ok_s} sandbox     : {sandbox_label()}")

    try:
        emb_ok = embeddings_available()
    except Exception:
        emb_ok = False
    print(f"    {ok_s if emb_ok else warn_s} embeddings  : "
          f"{'enabled (semantic_search active)' if emb_ok else 'disabled {SYM_EMDASH} set HERA_EMBED_URL'}")

    print(f"    {ok_s if WEB_ENABLED else warn_s} web search  : "
          f"{'on (' + SEARCH_PROVIDER + ')' if WEB_ENABLED else 'off (HERA_NO_WEB=1)'}")

    print(f"    {ok_s} MCP servers : {len(_mcp_clients)} connected")
    print(f"    {ok_s} memory      : {len(load_memory())} file(s) loaded")

    git_r = subprocess.run("git --version", shell=True, capture_output=True, text=True)
    print(f"    {ok_s if git_r.returncode == 0 else warn_s} git         : "
          f"{git_r.stdout.strip() if git_r.returncode == 0 else 'not found'}")

    gh_r = subprocess.run("gh --version", shell=True, capture_output=True, text=True)
    if gh_r.returncode == 0:
        print(f"    {ok_s} gh          : {gh_r.stdout.splitlines()[0].strip()} (needed for /pr)")
    else:
        print(f"    {warn_s} gh          : not found {SYM_EMDASH} needed for /pr")
        ans = input(f"    {CYAN}Install gh now? [y/N]{R} ").strip().lower()
        if ans == "y":
            result = tool_install("gh", "GitHub CLI for /pr command")
            sym = ok_s if not result.startswith("[error]") else fail_s
            print(f"    {sym} {result}")

    kb_path = os.path.join(CONFIG_DIR, "keybindings.json")
    if os.path.isfile(kb_path):
        nc = len(KEYBINDINGS.get("ctrl", {})); na = len(KEYBINDINGS.get("alt", {}))
        print(f"    {ok_s} keybindings : {nc} ctrl, {na} alt binding(s)")
    else:
        print(f"    {warn_s} keybindings : none {SYM_EMDASH} {kb_path}")
        ans = input(f"    {CYAN}Create default keybindings? [y/N]{R} ").strip().lower()
        if ans == "y":
            defaults = {"ctrl+r": "/review", "ctrl+p": "/pr",
                        "ctrl+g": "/diff",   "alt+d":  "/doctor"}
            try:
                os.makedirs(CONFIG_DIR, exist_ok=True)
                open(kb_path, "w").write(json.dumps(defaults, indent=2))
                print(f"    {fix_s} created {kb_path}")
                print(f"      {DIM}ctrl+r=/review  ctrl+p=/pr  ctrl+g=/diff  alt+d=/doctor{R}")
                print(f"      {DIM}restart hera to activate{R}")
            except OSError as e:
                print(f"    {fail_s} write failed: {e}")

    t = SESSION.get("total", 0)
    print(f"    {ok_s} session     : {t:,} tok, {SESSION.get('requests', 0)} requests{_cost_suffix()}")
    print()


# A per-turn override set from in-prompt keywords ("think hard", "ultrathink").
_TURN_THINK = None
_THINK_KEYWORD_RE = re.compile(
    r"\b(ultrathink|think (?:hard(?:er)?|deeply|step by step|carefully))\b", re.IGNORECASE)


def _keyword_think_level(text):
    """Map in-prompt thinking keywords to a level for this turn, or None."""
    return "hard" if _THINK_KEYWORD_RE.search(text or "") else None


def _think_payload():
    """Extra request fields for the effective thinking level. A per-turn keyword
    override (_TURN_THINK) wins over the configured THINK_LEVEL. off|normal|hard|max."""
    level = _TURN_THINK or THINK_LEVEL
    if level == "off":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if level in ("hard", "max"):
        return {"chat_template_kwargs": {"enable_thinking": True}}
    return {}  # 'normal' SYM_ARROW_R leave the server default


def _output_style_text():
    """Instruction block for the active output style (built-in or custom .md)."""
    if OUTPUT_STYLE in _OUTPUT_STYLES:
        return _OUTPUT_STYLES[OUTPUT_STYLE]
    p = os.path.join(CONFIG_DIR, "output-styles", f"{OUTPUT_STYLE}.md")
    if os.path.isfile(p):
        try:
            return open(p, encoding="utf-8", errors="replace").read()[:4000]
        except OSError:
            pass
    return ""


def output_styles_available():
    """Built-in + custom output style names."""
    names = list(_OUTPUT_STYLES)
    d = os.path.join(CONFIG_DIR, "output-styles")
    if os.path.isdir(d):
        names += [fn[:-3] for fn in sorted(os.listdir(d)) if fn.endswith(".md")]
    return names


def system_prompt():
    base = (
        f"You are {NAME}, an agentic coding assistant running in a terminal. "
        f"You operate in the working directory: {os.getcwd()} (OS: {sys.platform}). "
        "You have tools to list directories, find files by glob, search file contents by "
        "regex, index code definitions (symbols), read files, write files, edit files by "
        "exact string replacement, and run shell commands. A semantic_search tool may also "
        "be available for fuzzy 'where is the code thatSYM_ELLIPSIS' questions. "
        "Use glob/search/symbols to locate relevant code, then read files before changing "
        "anything; make edits with precise old_string/new_string. File edits are revertible "
        "by the user with /undo. For larger self-contained subtasks you may delegate to a "
        "focused sub-agent with the task tool. "
        "Keep prose short SYM_EMDASH act with tools rather than describing what you would do. "
        "When the task is complete, give a brief summary of what you changed."
        "\n\n"
        "PRECISION RULES SYM_EMDASH follow these without exception:\n"
        "1. NEVER state a fact about a file, config, command, or service without first reading "
        "the actual source. Read docker-compose.yml, config files, and code before claiming "
        "what they contain. Guessing is forbidden SYM_EMDASH if you don't know, use a tool to find out.\n"
        "2. ALWAYS give concrete, copy-pasteable solutions: exact shell commands with real flags, "
        "exact file paths, exact config keys and values. Never say 'something like' or "
        "'you might want to' SYM_EMDASH give the exact thing the user must run or change.\n"
        "3. COVER ALL AFFECTED SERVICES: when a task involves the running stack, read "
        "docker-compose.yml first and identify every service that is relevant to the problem "
        "(upstreams, downstreams, shared volumes, env vars). Address ALL of them SYM_EMDASH do not fix "
        "one service and silently leave a dependent service broken.\n"
        "4. VERIFY LIVE STATE BEFORE ANSWERING: for questions about what is running, broken, "
        "or configured, check the actual live state (docker ps, docker logs, curl health "
        "endpoints, cat config files) rather than inferring from memory or prior context.\n"
        "5. IF A COMMAND CAN FAIL, say exactly how to confirm it worked (the expected output "
        "or the verification command) so the user knows whether the fix took effect."
    )
    base += (" For any task with more than ~3 steps, call todo_write first to lay out the plan as "
             "a checklist, then update it as you go SYM_EMDASH mark each step completed and set the next one "
             "in_progress, keeping exactly one in_progress at a time. Skip it for trivial requests.")
    if EXTRA_DIRS:
        base += (" You may also read and write files in these additional trusted directories: "
                 + ", ".join(EXTRA_DIRS) + ".")
    if AUTO_VERIFY and not PLAN_MODE:
        base += (" ALWAYS VERIFY YOUR WORK: after writing or modifying code, actually run it before "
                 "saying you're done SYM_EMDASH run the project's tests/build/linter if present, or otherwise "
                 "execute the affected file or function (pytest, npm test/build, python <file>, "
                 "node <file>, go build ./SYM_ELLIPSIS, etc.). If verification fails, read the error, fix the "
                 "root cause, and re-run; repeat until it passes or you're genuinely blocked (then "
                 "explain what's blocking). When asked to run a project or codebase, get it actually "
                 "running and fix what breaks. Prefer the smallest relevant check.")
        base += _project_hint()
    if PLAN_MODE:
        base += (" PLAN MODE IS ON. Investigate with read-only tools only SYM_EMDASH do NOT modify files, "
                 "run state-changing commands, or install anything yet. When you have a concrete "
                 "plan, call the exit_plan_mode tool with a short numbered plan (markdown) and STOP. "
                 "The user approves there; on approval plan mode turns off and you implement. Use "
                 "the tool SYM_EMDASH don't just describe the plan in prose.")
    if WEB_ENABLED:
        base += (" You have live internet access. Whenever the answer depends on information you "
                 "don't already hold SYM_EMDASH current events, recent releases, library or API docs, exact "
                 "versions, an unfamiliar error message, anything time-sensitive SYM_EMDASH call web_search "
                 "on your own initiative rather than guessing, then web_fetch the most relevant "
                 "results to read their full text. Corroborate across two or more independent "
                 "sources before you commit to an answer. Then synthesize what you actually read "
                 "into a clear, direct answer in your own words SYM_EMDASH do not just paste snippets SYM_EMDASH and "
                 "cite the sources inline as [1], [2] with a short 'Sources:' list of the URLs at "
                 "the end. Treat freshly fetched pages as current ground truth over any older "
                 "assumption you have, and say so if sources disagree or you couldn't verify a claim.")
    if AUTO_INSTALL:
        base += (" When the task needs a command-line tool that isn't installed, call "
                 "install_tool with the program name and a short reason; the user approves, "
                 "then it's downloaded and you can use it via run_bash. You may also just run "
                 "a command SYM_EMDASH if it fails with 'command not found' the user is offered the "
                 "install and the command is retried automatically. Either way, don't give up "
                 "because a tool is missing.")
    style = _output_style_text()
    if style:
        base += f"\n\nOutput style ({OUTPUT_STYLE}): {style}"
    mem = load_memory()
    if mem:
        total = 0
        base += ("\n\nMemory & project context (follow these instructions and conventions; "
                 "more-specific scopes override broader ones):")
        for scope, path, body in mem:
            if total >= MEMORY_MAX_TOTAL:
                break
            body = body[:MEMORY_MAX_TOTAL - total]
            total += len(body)
            base += f"\n\n--- {scope} memory ({os.path.basename(path)}) ---\n{body}"
    git_ctx = _git_context()
    if git_ctx:
        base += f"\n\nGit context (live, auto-refreshed each session):\n{git_ctx}"
    return base


# ── @file mentions & context compaction ───────────────────────────────────────
def _image_data_url(path):
    """Read an image file and return a base64 `data:` URL, or None if unreadable
    or over MAX_IMAGE_BYTES."""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    try:
        if os.path.getsize(path) > MAX_IMAGE_BYTES:
            return None
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def expand_mentions(text, extra_images=None):
    """Resolve any @path the user references.

    Text files are inlined as before. Image @paths (and any pre-encoded
    `extra_images` SYM_EMDASH list of (name, data_url), used by the VS Code attach flow)
    become OpenAI `image_url` parts. Returns (content, names) where content is a
    plain string when no images are present, or a multimodal parts list when
    they are.
    """
    text_attached = []   # (name, text)
    images = []          # (name, data_url)
    for tok in re.findall(r"(?<!\S)@([^\s]+)", text):
        p = _resolve(tok)
        if not os.path.isfile(p):
            continue
        if tok.lower().endswith(IMAGE_EXTS):
            du = _image_data_url(p)
            if du:
                images.append((tok, du))
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                text_attached.append((tok, f.read()[:MAX_READ_BYTES]))
        except OSError:
            pass
    for name, du in (extra_images or []):
        if du:
            images.append((name, du))

    body = text
    if text_attached:
        blocks = "\n\n".join(f"--- {n} ---\n{c}" for n, c in text_attached)
        body = f"{text}\n\n[Attached files]\n{blocks}"
    names = [n for n, _ in text_attached] + [n for n, _ in images]
    if not images:
        return body, names
    parts = [{"type": "text", "text": body}]
    parts += [{"type": "image_url", "image_url": {"url": du}} for _, du in images]
    return parts, names


def compact_history(messages):
    """Replace the conversation with a model-written summary to free up context."""
    if len(messages) <= 2:
        return "nothing to compact yet"
    _run_hooks("PreCompact")
    convo = []
    for m in messages[1:]:
        c = _text_of(m.get("content"))
        if m.get("tool_calls"):
            c += " [tools: " + ", ".join(tc["function"]["name"] for tc in m["tool_calls"]) + "]"
        convo.append(f"{m.get('role')}: {c[:1500]}")
    prompt = ("Summarize this coding session concisely, preserving key decisions, file changes, "
              "and any open tasks so work can continue:\n\n" + "\n".join(convo))
    try:
        resp = requests.post(
            f"{API_URL}/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            timeout=300,
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"].get("content") or ""
    except Exception as exc:  # noqa: BLE001
        return f"[error] compact failed: {exc}"
    if not summary.strip():
        return "[error] compact produced no summary"
    messages[:] = [messages[0],
                   {"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"}]
    return f"compacted to a summary ({len(summary)} chars); history reset"


def _estimate_tokens(messages):
    """Rough token estimate (~4 chars/token) across the message list."""
    total = 0
    for m in messages:
        total += len(_text_of(m.get("content")))
        for tc in (m.get("tool_calls") or []):
            total += len(json.dumps(tc.get("function", {})))
    return total // 4


def _maybe_auto_compact(messages, emit=None):
    """Summarize history when it nears the context window. `emit(dict)` is the
    --serve event sink; when None we print to the terminal."""
    if CONTEXT_TOKENS <= 0 or AUTO_COMPACT_AT <= 0 or len(messages) <= 4:
        return
    # Use whichever is larger: the server's real last prompt size or our estimate.
    est = max(_estimate_tokens(messages), _LAST_PROMPT_TOKENS)
    if est < AUTO_COMPACT_AT * CONTEXT_TOKENS:
        return
    note = f"context ~{est} tok nearing the {CONTEXT_TOKENS} limit {SYM_EMDASH} auto-compacting{SYM_ELLIPSIS}"
    result_prefix = "auto-compact: "
    if emit:
        emit({"type": "info", "text": note})
    else:
        print(f"\n{DIM}  {note}{R}")
    result = compact_history(messages)
    if emit:
        emit({"type": "info", "text": result_prefix + result})
    else:
        print(f"{DIM}  {result_prefix}{result}{R}\n")


# ── Banner / help ──────────────────────────────────────────────────────────────
# (VERSION is defined once near the top of the file.)



def _short(path, n=40):
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path if len(path) <= n else "SYM_ELLIPSIS" + path[-(n - 1):]


def print_banner():
    host = API_URL.replace("http://", "").replace("https://", "").replace("/v1", "")
    fn, _ = load_project_context()
    shades = [SKY, SKY, TEAL, IND, IND]

    print()
    for sh, line in zip(shades, _WORDMARK):
        print(f"  {sh}{line}{R}")
    print(f"  {DIM}agentic coding CLI{R}  {GREY}{SYM_MIDDOT} v{VERSION} {SYM_MIDDOT} {MODEL}{R}\n")

    rule = f"  {GREY}{SYM_HLINE * 50}{R}"

    def row(label, value, vcolor=""):
        print(f"  {ACCENT}{SYM_ACCENT}{R} {DIM}{label:<8}{R}{vcolor}{value}{R}")

    print(rule)
    row("server", host)
    if USER_EMAIL or USER_NAME:
        row("account", whoami_label(), GREEN)
    row("cwd", _short(os.getcwd()))
    if YOLO:
        row("safety", "auto-approve (YOLO)", RED)
    elif AUTO_MODE != "read":
        desc = {"edit": "auto-approve reads + edits", "all": "auto-approve ALL (/auto off to stop)"}
        row("safety", f"auto: {AUTO_MODE} {SYM_EMDASH} {desc[AUTO_MODE]}", YELL)
    else:
        row("safety", "approval on edits & bash  (/auto for auto mode)")
    row("sandbox", sandbox_label())
    if ALLOW_PATTERNS:
        row("allow", f"{len(ALLOW_PATTERNS)} pattern(s)")
    if EXT_TOOLS:
        row("ext", f"{len(EXT_TOOLS)} mcp/custom tool(s)")
    if fn:
        row("context", fn, GREEN)
    row("tools", f"{len(TOOLS)} available")
    print(rule)
    print(f"  {DIM}type a task  {SYM_MIDDOT}  {R}{CYAN}@path{R}{DIM} to attach a file  {SYM_MIDDOT}  "
          f"press {R}{CYAN}/{R}{DIM} for commands{R}\n")


# ── Slash commands ────────────────────────────────────────────────────────────
# Single source of truth for the REPL's "/" commands. Drives Tab-completion,
# the pop-up recommendation menu, and /help so the three never drift apart.
# Each entry: (name, args-hint, one-line description).
SLASH_COMMANDS = [
    ("/help",      "",          "show this list of commands"),
    ("/skills",    "[id]",      "list shared skills, or show one in detail"),
    ("/undo",      "",          "revert the last file write/edit I made"),
    ("/rewind",    "[n]",       "restore conversation AND files to an earlier turn"),
    ("/context",   "",          "show how the context window is being used"),
    ("/memory",    "",          "show loaded memory files (add with # <fact>)"),
    ("/init",      "",          "analyze the project and generate a HERA.md"),
    ("/export",    "[path]",    "save the conversation to Markdown (or .json)"),
    ("/plugins",   "[install]", "list plugins, browse the marketplace, or install one"),
    ("/doctor",    "",          "self-update + stack health check + CLI diagnostics/auto-fix"),
    ("/review",    "",          "code-review the current git diff"),
    ("/pr",        "[title]",   "create a GitHub PR from the current branch (uses gh)"),
    ("/diff",      "",          "show the working-tree git diff"),
    ("/compact",   "",          "summarize the conversation to free up context"),
    ("/tokens",    "",          "show token usage (and cost, if priced) this session"),
    ("/plan",      "",          "toggle plan mode (investigate & propose before editing)"),
    ("/auto",      "[mode]",    "auto-approve level for this project: read / edit / all / off"),
    ("/todos",     "",          "show the current to-do checklist"),
    ("/tools",     "",          "list the tools I can use"),
    ("/allow",     "[pattern]", "list run_bash allow patterns, or add one"),
    ("/sandbox",   "",          "show the run_bash sandbox status"),
    ("/add-dir",   "[path]",    "grant access to an extra directory beyond cwd"),
    ("/agents",    "",          "list named sub-agents"),
    ("/mcp",       "",          "list connected MCP servers"),
    ("/permissions", "[rule]",  "list permission rules, or add one (deny/ask/allow)"),
    ("/config",    "",          "show effective settings"),
    ("/sessions",  "",          "list saved conversations (by their first message)"),
    ("/resume",    "",          "pick a past conversation to resume (by its first message)"),
    ("/reasoning", "",          "toggle streaming of my thinking"),
    ("/vim",       "",          "toggle vim keybindings in the prompt editor"),
    ("/think",     "[level]",   "thinking budget: off / normal / hard"),
    ("/output-style", "[name]", "answer style: default / concise / explanatory / learning"),
    ("/statusline", "",         "show/set the status line (set to 'builtin' for built-in bar)"),
    ("/cwd",       "",          "show the working directory"),
    ("/new",       "",          "save current and start a fresh session"),
    ("/clear",     "",          "same as /new (fresh conversation)"),
    ("/logout",    "",          "sign out and switch to a different API key"),
    ("/exit",      "",          "quit  (Ctrl-C or Ctrl-D also work)"),
]
_SLASH_HELP = {name: (args, desc) for name, args, desc in SLASH_COMMANDS}


def _slash_row(name, args, desc):
    label = f"{name} {args}".rstrip()
    return f"  {CYAN}{label:<24}{R}{DIM}{desc}{R}"


def print_slash_menu(typed=""):
    """Print the available slash commands, Claude-Code style.

    Filters by what the user has typed so far; a bare "/" lists everything.
    Used as the fallback when someone sends a line that starts with "/" but
    isn't a recognized command.
    """
    prefix = (typed.split() or ["/"])[0].lower()
    matches = [c for c in SLASH_COMMANDS if c[0].startswith(prefix)]
    if typed and prefix != "/" and not matches:
        print(f"\n{DIM}unknown command {prefix!r}. available commands:{R}")
        matches = SLASH_COMMANDS
    else:
        print()
    for name, args, desc in matches:
        print(_slash_row(name, args, desc))
    print(f"\n  {DIM}type {R}{CYAN}/{R}{DIM} then {R}{CYAN}Tab{R}{DIM} to autocomplete{R}\n")


# ── Tab-completion for slash commands ─────────────────────────────────────────
def _slash_completer(text, state):
    """readline completer: completes "/..." against SLASH_COMMANDS."""
    if not text.startswith("/"):
        return None
    matches = [name + " " for name, *_ in SLASH_COMMANDS if name.startswith(text.lower())]
    return matches[state] if state < len(matches) else None


def _slash_display(substitution, matches, longest):
    """Show each completion with its description rather than a bare grid."""
    print()
    for m in matches:
        name = m.strip()
        args, desc = _SLASH_HELP.get(name, ("", ""))
        print(_slash_row(name, args, desc))
    try:
        readline.redisplay()
    except Exception:
        pass


def setup_completion():
    """Wire up Tab-completion of slash commands in the interactive REPL."""
    try:
        import readline
    except ImportError:
        return
    readline.set_completer(_slash_completer)
    # Keep the whole "/word" as one token (the default delims split on "/").
    readline.set_completer_delims(" \t\n")
    try:
        readline.set_completion_display_matches_hook(_slash_display)
    except (AttributeError, NotImplementedError):
        pass
    if "libedit" in (getattr(readline, "__doc__", "") or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
        # Single Tab lists everything — no "display all 14 possibilities? (y/n)".
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("set completion-query-items 200")


# ── Raw-mode prompt with a live slash-command dropdown ────────────────────────
# A self-contained line editor (no third-party deps). It mirrors Claude Code:
# the moment the line starts with "/", a filtering menu pops up under the
# prompt — arrow keys move the highlight, Tab/Enter accept it, typing filters,
# Backspace past the "/" closes it. Falls back to input() when stdin isn't a
# TTY (pipes, the --serve path) or on platforms without termios.
INPUT_HISTORY = []
_MENU_LABEL_W = 22  # column width reserved for the "/cmd [args]" label


def _menu_active(buf):
    """The dropdown shows while the buffer is a bare, still-unfinished /command."""
    return re.fullmatch(r"/[A-Za-z-]*", buf) is not None


def _menu_matches(buf):
    return [c for c in SLASH_COMMANDS if c[0].startswith(buf.lower())]


# ── Vim keybindings (pure state machine; positions are insert-style 0..len) ────
def _vim_next_word(buf, pos):
    n = len(buf)
    i = pos
    if i >= n:
        return n
    if buf[i].isspace():
        while i < n and buf[i].isspace():
            i += 1
    else:
        while i < n and not buf[i].isspace():
            i += 1
        while i < n and buf[i].isspace():
            i += 1
    return i


def _vim_prev_word(buf, pos):
    i = pos - 1
    while i > 0 and buf[i].isspace():
        i -= 1
    while i > 0 and not buf[i - 1].isspace():
        i -= 1
    return max(i, 0)


def _vim_word_end(buf, pos):
    n = len(buf)
    i = pos + 1
    while i < n and buf[i].isspace():
        i += 1
    while i + 1 < n and not buf[i + 1].isspace():
        i += 1
    return min(i, n)


def vim_normal_key(buf, pos, key, pending=""):
    """Apply one NORMAL-mode key. Returns (buf, pos, mode, pending, action) where
    mode is 'normal' or 'insert' and action is None or 'submit'. Pure/testable."""
    # An operator (d/c) is pending — this key is its motion.
    if pending in ("d", "c"):
        to_insert = pending == "c"
        if key == pending:                       # dd / cc SYM_ARROW_R whole line
            return ("", 0, "insert" if to_insert else "normal", "", None)
        end = {"w": _vim_next_word(buf, pos), "b": _vim_prev_word(buf, pos),
               "e": _vim_word_end(buf, pos), "$": len(buf), "0": 0,
               "l": min(len(buf), pos + 1), "h": max(0, pos - 1)}.get(key)
        if end is None:
            return (buf, pos, "normal", "", None)  # unknown motion cancels op
        lo, hi = sorted((pos, end))
        return (buf[:lo] + buf[hi:], lo, "insert" if to_insert else "normal", "", None)
    if key in ("d", "c"):
        return (buf, pos, "normal", key, None)
    if key == "ENTER":
        return (buf, pos, "normal", "", "submit")
    if key == "i":
        return (buf, pos, "insert", "", None)
    if key == "a":
        return (buf, min(len(buf), pos + 1), "insert", "", None)
    if key == "I":
        return (buf, 0, "insert", "", None)
    if key == "A":
        return (buf, len(buf), "insert", "", None)
    if key == "x":
        return (buf[:pos] + buf[pos + 1:], pos, "normal", "", None)
    if key == "D":
        return (buf[:pos], pos, "normal", "", None)
    if key == "C":
        return (buf[:pos], pos, "insert", "", None)
    if key == "h":
        return (buf, max(0, pos - 1), "normal", "", None)
    if key == "l":
        return (buf, min(len(buf), pos + 1), "normal", "", None)
    if key == "0":
        return (buf, 0, "normal", "", None)
    if key == "$":
        return (buf, len(buf), "normal", "", None)
    if key == "w":
        return (buf, _vim_next_word(buf, pos), "normal", "", None)
    if key == "b":
        return (buf, _vim_prev_word(buf, pos), "normal", "", None)
    if key == "e":
        return (buf, _vim_word_end(buf, pos), "normal", "", None)
    return (buf, pos, "normal", "", None)  # ignore everything else


class RawLineReader:
    """One-line raw-mode editor with the slash-command dropdown."""

    def __init__(self, prompt):
        self.prompt = prompt
        self.pw = len(_strip_ansi(prompt))
        self.fd = sys.stdin.fileno()
        self.buf = ""
        self.pos = 0          # cursor index within buf
        self.sel = 0          # highlighted menu row
        self.hist = len(INPUT_HISTORY)  # index into history (== len means "current")
        self.stash = ""       # buffer stashed while browsing history
        self.prev_row = 0     # screen-row offset of the cursor after last render
        self.vim_mode = "insert"   # when VIM_MODE: 'insert' or 'normal'
        self.vim_pending = ""      # a pending vim operator (d/c)

    # — key decoding ————————————————————————————————————————————————
    def _key(self):
        b = os.read(self.fd, 1)
        if not b:
            return "EOF"
        c = b[0]
        simple = {0x03: "CTRL_C", 0x04: "CTRL_D", 0x01: "HOME", 0x05: "END",
                  0x0b: "CTRL_K", 0x15: "CTRL_U", 0x17: "CTRL_W",
                  0x0d: "ENTER", 0x0a: "ENTER", 0x09: "TAB",
                  0x7f: "BACKSPACE", 0x08: "BACKSPACE"}
        if c in simple:
            return simple[c]
        if c == 0x1b:
            return self._escape()
        if c < 0x20:
            # Check custom ctrl keybindings before discarding.
            if c in _KB_CTRL:
                return f"KEYBIND:{_KB_CTRL[c]}"
            return None  # ignore other control bytes
        # printable / start of a UTF-8 sequence
        n = 1 if c < 0x80 else (2 if c >> 5 == 0b110 else (3 if c >> 4 == 0b1110 else 4))
        if n > 1:
            b += os.read(self.fd, n - 1)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _escape(self):
        r, _, _ = select.select([self.fd], [], [], 0.03)
        if not r:
            return "ESC"
        nxt = os.read(self.fd, 1)
        # Alt+letter: ESC followed by a single letter (not [ or O).
        if nxt not in (b"[", b"O"):
            ch = nxt.decode("latin-1", errors="replace").lower()
            if ch in _KB_ALT:
                return f"KEYBIND:{_KB_ALT[ch]}"
            return "ESC"
        final = os.read(self.fd, 1)
        arrows = {b"A": "UP", b"B": "DOWN", b"C": "RIGHT", b"D": "LEFT",
                  b"H": "HOME", b"F": "END"}
        if final in arrows:
            return arrows[final]
        if final in (b"1", b"3", b"4", b"7", b"8"):
            while True:  # consume the trailing "~"
                t = os.read(self.fd, 1)
                if not t or t == b"~":
                    break
            return {b"1": "HOME", b"7": "HOME", b"4": "END", b"8": "END",
                    b"3": "DELETE"}.get(final, "ESC")
        return "ESC"

    # — rendering ————————————————————————————————————————————————————
    def _menu_lines(self, matches, width):
        rows = []
        desc_budget = max(0, width - 3 - _MENU_LABEL_W - 1)
        for i, (name, args, desc) in enumerate(matches):
            label = f"{name} {args}".rstrip()
            descT = desc[:desc_budget]
            if i == self.sel:
                vis = f"{label:<{_MENU_LABEL_W}} {descT}"
                vis = vis[:width - 3]
                rows.append(f"{REV} {SYM_PROMPT} {vis}{' ' * (width - 3 - len(vis))}{R}")
            else:
                rows.append(f"   {CYAN}{label:<{_MENU_LABEL_W}}{R} {DIM}{descT}{R}")
        return rows

    def _render(self, menu_rows):
        width = max(20, shutil.get_terminal_size((80, 24)).columns)
        out = []
        if self.prev_row:
            out.append(f"\033[{self.prev_row}A")
        out.append("\r\033[J")                 # to prompt row, clear everything below
        out.append(self.prompt + self.buf)
        for row in menu_rows:
            out.append("\r\n" + row)
        end_cells = self.pw + len(self.buf)
        cur_phys = end_cells // width + len(menu_rows)
        tgt_cells = self.pw + self.pos
        tgt_row, tgt_col = divmod(tgt_cells, width)
        up = cur_phys - tgt_row
        if up > 0:
            out.append(f"\033[{up}A")
        out.append("\r")
        if tgt_col:
            out.append(f"\033[{tgt_col}C")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self.prev_row = tgt_row

    def _refresh(self):
        if _menu_active(self.buf):
            matches = _menu_matches(self.buf)
            self.sel = min(self.sel, len(matches) - 1) if matches else 0
            self._render(self._menu_lines(matches, max(20, shutil.get_terminal_size((80, 24)).columns)))
        else:
            self._render([])

    def _close(self):
        """Erase the menu, drop to a fresh line below the (full) buffer."""
        self.pos = len(self.buf)
        self._render([])
        sys.stdout.write("\r\n")
        sys.stdout.flush()

    # — main loop ————————————————————————————————————————————————————
    def read(self):
        old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        try:
            self._refresh()
            while True:
                key = self._key()
                active = _menu_active(self.buf)
                matches = _menu_matches(self.buf) if active else []

                # Vim NORMAL mode: route printable keys through the vim engine.
                if (VIM_MODE and self.vim_mode == "normal" and not (active and matches)
                        and isinstance(key, str) and len(key) == 1 and key >= " "):
                    self.buf, self.pos, self.vim_mode, self.vim_pending, act = \
                        vim_normal_key(self.buf, self.pos, key, self.vim_pending)
                    if act == "submit":
                        self._close(); return self.buf
                    self.sel = 0; self._refresh(); continue

                if isinstance(key, str) and key.startswith("KEYBIND:"):
                    self.buf = key[len("KEYBIND:"):]
                    self._close()
                    return self.buf

                if key == "ENTER":
                    if active and matches:
                        name, args, _ = matches[self.sel]
                        if args:                      # needs an argument: fill, keep editing
                            self.buf, self.pos, self.sel = name + " ", len(name) + 1, 0
                            self._refresh()
                            continue
                        self.buf = name               # no-arg command: submit it
                    self._close()
                    return self.buf
                if key == "TAB":
                    if active and matches:
                        name = matches[self.sel][0]
                        self.buf, self.pos, self.sel = name + " ", len(name) + 1, 0
                        self._refresh()
                    continue
                if key == "UP":
                    if active and matches:
                        self.sel = max(0, self.sel - 1)
                    else:
                        self._history(-1)
                    self._refresh(); continue
                if key == "DOWN":
                    if active and matches:
                        self.sel = min(len(matches) - 1, self.sel + 1)
                    else:
                        self._history(1)
                    self._refresh(); continue
                if key == "LEFT":
                    self.pos = max(0, self.pos - 1); self._refresh(); continue
                if key == "RIGHT":
                    self.pos = min(len(self.buf), self.pos + 1); self._refresh(); continue
                if key == "HOME":
                    self.pos = 0; self._refresh(); continue
                if key == "END":
                    self.pos = len(self.buf); self._refresh(); continue
                if key == "BACKSPACE":
                    if self.pos:
                        self.buf = self.buf[:self.pos - 1] + self.buf[self.pos:]
                        self.pos -= 1; self.sel = 0
                    self._refresh(); continue
                if key == "DELETE":
                    if self.pos < len(self.buf):
                        self.buf = self.buf[:self.pos] + self.buf[self.pos + 1:]
                        self.sel = 0
                    self._refresh(); continue
                if key == "CTRL_U":
                    self.buf = self.buf[self.pos:]; self.pos = 0; self.sel = 0
                    self._refresh(); continue
                if key == "CTRL_K":
                    self.buf = self.buf[:self.pos]; self.sel = 0; self._refresh(); continue
                if key == "CTRL_W":
                    self._delete_word(); self._refresh(); continue
                if key == "ESC":
                    if VIM_MODE:               # ESC SYM_ARROW_R NORMAL mode (vim), keep the line
                        self.vim_mode = "normal"; self.vim_pending = ""
                        self.pos = max(0, self.pos - 1); self.sel = 0
                        self._refresh(); continue
                    self.buf = ""; self.pos = 0; self.sel = 0; self._refresh(); continue
                if key == "CTRL_C":
                    self._close(); raise KeyboardInterrupt
                if key in ("CTRL_D", "EOF"):
                    if not self.buf:
                        self._close(); raise EOFError
                    if self.pos < len(self.buf):       # otherwise act as delete-forward
                        self.buf = self.buf[:self.pos] + self.buf[self.pos + 1:]
                        self.sel = 0; self._refresh()
                    continue
                if isinstance(key, str) and key >= " ":
                    self.buf = self.buf[:self.pos] + key + self.buf[self.pos:]
                    self.pos += len(key); self.sel = 0; self._refresh()
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old)

    def _history(self, step):
        if step < 0:
            if self.hist > 0:
                if self.hist == len(INPUT_HISTORY):
                    self.stash = self.buf
                self.hist -= 1
                self.buf = INPUT_HISTORY[self.hist]
        else:
            if self.hist < len(INPUT_HISTORY):
                self.hist += 1
                self.buf = (INPUT_HISTORY[self.hist] if self.hist < len(INPUT_HISTORY)
                            else self.stash)
        self.pos = len(self.buf); self.sel = 0

    def _delete_word(self):
        i = self.pos
        while i > 0 and self.buf[i - 1] == " ":
            i -= 1
        while i > 0 and self.buf[i - 1] != " ":
            i -= 1
        self.buf = self.buf[:i] + self.buf[self.pos:]
        self.pos = i; self.sel = 0


def read_line(prompt):
    """Read one line of input, with the live slash-menu when on a real TTY."""
    use_raw = (termios is not None and sys.stdin.isatty() and sys.stdout.isatty())
    if not use_raw:
        return input(prompt)
    try:
        line = RawLineReader(prompt).read()
    except termios.error:
        return input(prompt)
    if line.strip() and (not INPUT_HISTORY or INPUT_HISTORY[-1] != line):
        INPUT_HISTORY.append(line)
    return line


def print_help():
    print(f"\n{DIM}"
          f"  Ask me to build, fix, explain, or refactor code. I work in the\n"
          f"  current directory and ask before editing files or running shell\n"
          f"  commands (unless HERA_YOLO=1). Attach a file or image with @path\n"
          f"  (e.g. @screenshot.png). At an approval prompt, choose [t] to type an\n"
          f"  instruction instead of yes/no. Press {R}{CYAN}ESC{R}{DIM} while I'm working to interrupt."
          f"{R}\n")
    for name, args, desc in SLASH_COMMANDS:
        print(_slash_row(name, args, desc))
    print(f"\n{DIM}"
          f"  Tip: press {R}{CYAN}/{R}{DIM} to open this menu inline {SYM_EMDASH} {R}{CYAN}{SYM_ARROW_U}{R}{DIM}/{R}{CYAN}{SYM_ARROW_D}{R}{DIM} to\n"
          f"  pick, {R}{CYAN}Tab{R}{DIM} or {R}{CYAN}Enter{R}{DIM} to accept, keep typing to filter.\n\n"
          f"  Start with --resume [ID] / --continue to pick up a past session,\n"
          f"  or --list-sessions to see them.{R}\n")


# ── Headless JSON mode (for the VS Code webview) ──────────────────────────────
# `hera --serve` speaks newline-delimited JSON on stdin/stdout so a GUI can drive
# the full agent. stdout carries ONLY JSON events; logs go to stderr.
#
#   in : {"type":"prompt","text":...,"images":[dataURL,…]}
#        | {"type":"approval","decision":"y|a|p|n","feedback":"…"}
#        | {"type":"plan","on":bool} | {"type":"plan_decision","decision":"yes|auto|no","feedback":"…"}
#        | {"type":"auto","mode":…} | {"type":"logout"}
#        | {"type":"interrupt"} | {"type":"undo"} | {"type":"clear"} | {"type":"exit"}
#   out: ready | reasoning | token | narration | tool_start | proposed_diff
#        | approval_request | plan_review | tool_end | turn_end | todos | suggestions
#        | auto_mode | logged_out | info | error
#
# A single reader thread (_serve_input_thread) demultiplexes stdin so the editor
# can send an `interrupt` or an `approval` *while* a turn is streaming: approvals
# go to _APPROVAL_Q, an interrupt sets _INTERRUPT, everything else to _MAIN_Q.
_MAIN_Q = queue.Queue()
_APPROVAL_Q = queue.Queue()
_SERVE_CLOSED = threading.Event()


def _default_emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# Swappable so headless `-p` (print) mode can capture events instead of writing
# the raw serve JSON stream. serve mode uses the default sink.
_EMIT_SINK = _default_emit
# When True (headless -p without a TTY), tool approvals can't prompt — they're
# auto-denied unless YOLO / auto-mode / allowlist already granted them.
_NONINTERACTIVE = False


def _emit(obj):
    _EMIT_SINK(obj)


# NOTE: stdin is consumed exclusively by _serve_input_thread below — nothing in
# serve mode reads sys.stdin directly (that would race the reader thread).
def _serve_input_thread():
    while True:
        line = sys.stdin.readline()
        if not line:                       # stdin closed SYM_ARROW_R tell both consumers
            _SERVE_CLOSED.set()
            _MAIN_Q.put(None)
            _APPROVAL_Q.put(None)
            return
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = msg.get("type")
        if t == "interrupt":
            _INTERRUPT.set()
        elif t in ("approval", "plan_decision"):
            _APPROVAL_Q.put(msg)
        elif t == "plan":
            global PLAN_MODE
            PLAN_MODE = bool(msg.get("on", not PLAN_MODE))
            _emit({"type": "info", "text": f"plan mode {'on' if PLAN_MODE else 'off'}"})
        elif t == "auto":
            global AUTO_MODE
            want = str(msg.get("mode", "read")).lower()
            want = {"off": "read", "write": "edit"}.get(want, want)
            if want in _AUTO_LEVELS:
                AUTO_MODE = want
                _save_auto_mode(want)
                _emit({"type": "info", "text": f"auto mode {SYM_ARROW_R} {want}"})
                _emit({"type": "auto_mode", "mode": want})
        elif t == "logout":
            logout()
            _emit({"type": "info", "text": "logged out SYM_EMDASH set hera.apiKey to a new key and reload."})
            _emit({"type": "logged_out"})
        else:
            _MAIN_Q.put(msg)


def _serve_anthropic(messages):
    """Anthropic-provider turn for --serve: non-streaming, emits the answer once."""
    base, headers = _anthropic_headers()
    system, amsgs = _to_anthropic_messages(messages)
    body = {"model": MODEL, "max_tokens": MAX_OUTPUT_TOKENS, "messages": amsgs,
            "tools": _anthropic_tools(TOOL_SCHEMAS)}
    if system:
        body["system"] = system
    try:
        resp = requests.post(f"{base}/v1/messages", json=body, headers=headers, timeout=600)
        resp.raise_for_status()
        result = _parse_anthropic_response(resp.json())
    except requests.exceptions.RequestException as exc:
        _emit({"type": "error", "message": str(exc)})
        return None
    if result["content"]:
        _emit({"type": "token", "delta": result["content"]})
    return result


def _serve_stream(messages):
    global _VISION_WARNED
    if _is_anthropic():
        return _serve_anthropic(messages)
    url, model, send_messages, downgraded = _select_endpoint(messages)
    if downgraded and not _VISION_WARNED:
        _VISION_WARNED = True
        _emit({"type": "info", "text": "image attached, but the current model is "
               "text-only SYM_EMDASH set HERA_VISION_URL to enable vision."})
    resp = None
    for compacted in (False, True):
        try:
            resp = requests.post(
                f"{url}/chat/completions",
                json={"model": model, "messages": send_messages, "tools": TOOL_SCHEMAS,
                      "tool_choice": "auto", "stream": True,
                      "stream_options": {"include_usage": True}, **_think_payload()},
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                stream=True, timeout=600,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            # Self-heal a context-overflow 400: summarize and retry once.
            if (code == 400 and _is_context_overflow(exc)
                    and not compacted and len(messages) > 2):
                _emit({"type": "info", "text": "context window full SYM_EMDASH compacting and retryingSYM_ELLIPSIS"})
                compact_history(messages)
                url, model, send_messages, downgraded = _select_endpoint(messages)
                continue
            _emit({"type": "error", "message": str(exc)})
            return None
        except requests.exceptions.RequestException as exc:
            _emit({"type": "error", "message": str(exc)})
            return None
    if resp is None:
        return None

    content, tool_calls, usage, finish = [], {}, None, None
    for raw in resp.iter_lines():
        if _INTERRUPT.is_set():
            finish = "interrupted"
            resp.close()
            break
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
        if delta.get("reasoning_content"):
            _emit({"type": "reasoning", "delta": delta["reasoning_content"]})
        if delta.get("content"):
            content.append(delta["content"])
            _emit({"type": "token", "delta": delta["content"]})
        for tc in delta.get("tool_calls", []):
            slot = tool_calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function", {})
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
    final_content = "".join(content)
    structured = [tool_calls[i] for i in sorted(tool_calls)]
    if not structured:  # recover tool calls the server left in the text
        final_content, recovered = _extract_text_tool_calls(final_content)
        if recovered:
            structured = recovered
            _emit({"type": "info",
                   "text": f"recovered {len(recovered)} tool call(s) from text"})
    return {"content": final_content, "finish_reason": finish,
            "tool_calls": structured, "usage": usage}


def _serve_approve(name, args):
    # Same gating as the interactive approve(): plan mode, config permissions,
    # PreToolUse hooks, then auto mode / YOLO / allowlist — before prompting.
    if PLAN_MODE and name in SIDE_EFFECTS:
        return ("plan mode is on SYM_EMDASH investigate read-only, then call exit_plan_mode with your "
                "plan for the user to approve. Modifying tools stay disabled until then.")
    decision = _perm_decision(name, args)
    if decision == "deny":
        return "denied by a permission rule"
    blocked = _run_hooks("PreToolUse", name, args)
    if blocked:
        return blocked
    if decision == "allow":
        return True
    if decision != "ask":
        if YOLO or AUTO_MODE == "all":
            return True
        if AUTO_MODE == "edit" and name in _EDIT_TOOLS:
            return True
        if name not in SIDE_EFFECTS or name in _always_ok:
            return True
    if name == "run_bash" and decision != "ask" and bash_allowed(args.get("command", "")):
        return True
    if _NONINTERACTIVE:  # headless -p: nothing to prompt, so deny safely
        return (f"requires approval ({name}); run with --yolo or set auto mode "
                f"to allow it in headless mode")
    _emit({"type": "approval_request", "name": name,
           "preview": _strip_ansi(_preview_call(name, args)),
           "command": args.get("command", "")})
    while True:
        msg = _APPROVAL_Q.get()
        if msg is None:
            return "aborted (input closed)"
        if msg.get("type") == "approval":
            d = msg.get("decision", "n")
            fb = (msg.get("feedback") or "").strip()
            if d not in ("y", "a", "p") and fb:
                return fb  # typed instruction SYM_ARROW_R fed straight back to the model
            if d == "a" and name == "run_bash":
                ALLOW_PATTERNS.append(" ".join(args.get("command", "").split()))
            elif d == "a":
                _always_ok.add(name)
            elif d == "p" and name == "run_bash" and args.get("command", "").split():
                ALLOW_PATTERNS.append(args["command"].split()[0] + " *")
            return True if d in ("y", "a", "p") else "user declined"


def _serve_install_approver(program, plan):
    """IDE approval for the reactive run_bash SYM_ARROW_R missing-binary install offer.

    Emits the same approval_request event the editor already renders as buttons,
    then blocks on the editor's decision (read from the JSON stdin stream)."""
    _emit({"type": "approval_request", "name": "install_tool",
           "preview": f"install {program}  {SYM_EMDASH} required by the last command\n    $ {plan}",
           "command": ""})
    while True:
        msg = _APPROVAL_Q.get()
        if msg is None:
            return False
        if msg.get("type") == "approval":
            return msg.get("decision", "n") in ("y", "a", "p")


def _serve_plan_approver(plan):
    """IDE approval for plan mode: show the plan, await the editor's decision."""
    _emit({"type": "plan_review", "plan": plan})
    while True:
        msg = _APPROVAL_Q.get()
        if msg is None:
            return "no", ""
        if msg.get("type") == "plan_decision":
            d = msg.get("decision", "no")
            return (d if d in ("yes", "auto", "no") else "no"), (msg.get("feedback") or "")


def _serve_exec(c):
    name = c["name"]
    try:
        args = json.loads(c["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    _emit({"type": "tool_start", "name": name, "narration": _narrate(name, args),
           "preview": _strip_ansi(_preview_call(name, args))})
    if name not in TOOLS:
        out = f"[error] unknown tool: {name}"
        _emit({"type": "tool_end", "name": name, "error": True, "output": out})
        return out
    # Show the change the editor *will* make before it's applied — compute the
    # proposed before/after without touching the file.
    if name in ("write_file", "edit_file"):
        path = args.get("path", "")
        existed, before = _snapshot(path)
        after = None
        if name == "write_file":
            after = args.get("content", "")
        else:
            old_s, new_s = args.get("old_string", ""), args.get("new_string", "")
            if existed and before is not None and old_s in before:
                after = (before.replace(old_s, new_s) if args.get("replace_all")
                         else before.replace(old_s, new_s, 1))
        if after is not None:
            _emit({"type": "proposed_diff", "path": _resolve(path),
                   "before": (before or "")[:200_000], "after": after[:200_000]})
    verdict = _serve_approve(name, args)
    if verdict is not True:
        out = f"[denied] {verdict}"
        _emit({"type": "tool_end", "name": name, "error": True, "output": out})
        return out
    snap = _snapshot(args.get("path", "")) if name in _EDIT_TOOLS else None
    try:
        out = TOOLS[name](**args)
    except TypeError as exc:
        msg = str(exc).replace(f"tool_{name}()", f"{name}()")
        out = f"[error] bad arguments: {msg}"
    except Exception as exc:  # noqa: BLE001
        out = f"[error] {type(exc).__name__}: {exc}"
    is_err = str(out).startswith("[error]")
    event = {"type": "tool_end", "name": name, "error": is_err, "output": out[:600]}
    if snap is not None and not is_err:
        push_checkpoint(args.get("path", ""), snap, name)
        # Include before/after (capped) so the editor can show a native diff.
        after_exist, after = _snapshot(args.get("path", ""))
        event["diff"] = {
            "path": _resolve(args.get("path", "")),
            "before": (snap[1] or "")[:200_000],
            "after": (after or "")[:200_000],
        }
    _emit(event)
    # Push the refreshed checklist to the editor after a to-do update.
    if name == "todo_write" and not is_err:
        _emit({"type": "todos", "items": list(TODOS)})
    _run_hooks("PostToolUse", name, args, out)
    if MAX_TOOL_OUTPUT and len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + f"\n{SYM_ELLIPSIS}[truncated, {len(out)} chars total]"
    return out


def _serve_run(messages):
    turn = 0
    did_work = False
    edited_code = ran_command = verified = False
    edit_attempted = edit_nudged = False
    last_user = next((_text_of(m.get("content")) for m in reversed(messages)
                      if m.get("role") == "user"), "")
    verify_requested = _wants_run_verification(last_user)
    change_requested = _wants_code_change(last_user)
    globals()["_TURN_THINK"] = _keyword_think_level(last_user)
    _maybe_auto_compact(messages, emit=_emit)
    serve_step = 0
    while True:
        if MAX_STEPS and serve_step >= MAX_STEPS:
            _emit({"type": "turn_end", "content": f"[stopped: hit MAX_STEPS={MAX_STEPS}]",
                   "turn_tokens": turn, "session_tokens": dict(SESSION)})
            return
        serve_step += 1
        res = _serve_stream(messages)
        if res is None:
            return
        turn += _account(res.get("usage"))
        if res.get("finish_reason") == "interrupted":
            messages.append({"role": "assistant",
                             "content": res["content"] or "(interrupted)"})
            _emit({"type": "turn_end", "content": res["content"] or "",
                   "interrupted": True, "turn_tokens": turn,
                   "session_tokens": dict(SESSION)})
            return
        calls = res["tool_calls"]
        am = {"role": "assistant", "content": res["content"] or ""}
        if calls:
            am["tool_calls"] = [{"id": x["id"], "type": "function",
                                 "function": {"name": x["name"],
                                              "arguments": _normalize_tool_args(x["arguments"])}}
                                for x in calls]
        messages.append(am)
        if not calls:
            # Apply-your-edit loop (before turn_end, so the panel stays busy):
            # the user asked for a change but the model never emitted an edit/
            # write call — nudge once to actually apply it.
            if (AUTO_VERIFY and change_requested and not edit_attempted
                    and not edit_nudged and not PLAN_MODE):
                edit_nudged = True
                messages.append({"role": "user", "content": _EDIT_NUDGE})
                _emit({"type": "info", "text": "auto-apply: applying the change you describedSYM_ELLIPSIS"})
                continue
            # Verify-your-work loop (before turn_end, so the panel stays busy
            # through the check): code changed (or the user asked to run the
            # project) but nothing ran → nudge once.
            if (AUTO_VERIFY and (edited_code or verify_requested) and not ran_command
                    and not verified and not PLAN_MODE):
                verified = True
                edited_code = False
                messages.append({"role": "user", "content": _verify_nudge()})
                _emit({"type": "info", "text": "auto-verify: running it to make sure it worksSYM_ELLIPSIS"})
                continue
            _emit({"type": "turn_end", "content": res["content"] or "",
                   "turn_tokens": turn, "session_tokens": dict(SESSION),
                   "cost": _session_cost()})
            _run_hooks("Stop")
            # End-of-task next-step tips for the editor UI (same feature the
            # interactive CLI prints). Emitted after turn_end so the panel has
            # already unblocked; best-effort, so a failure is silently dropped.
            if did_work:
                items = _generate_suggestions(messages)
                if items:
                    _emit({"type": "suggestions", "items": items})
            return
        did_work = True
        for c in calls:
            out = _serve_exec(c)
            if c["name"] == "run_bash":
                ran_command = True
            elif c["name"] in _EDIT_TOOLS:
                edit_attempted = True
                try:
                    cargs = json.loads(c["arguments"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    cargs = {}
                if _is_code_file(cargs.get("path", "")):
                    edited_code = True
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})


def print_main(prompt_text, output_format="text", max_turns=None):
    """Headless one-shot run (Claude-Code `-p`). Reads `prompt_text` (or stdin),
    runs a single agentic task non-interactively, and prints the result in
    `output_format`: text (final answer), json (one result object), or
    stream-json (one JSON event per line). Returns a process exit code."""
    global _EMIT_SINK, _NONINTERACTIVE, _PLAN_APPROVER
    if not API_URL:
        print("[error] no endpoint set SYM_EMDASH export HERA_API_URL / HERA_API_KEY", file=sys.stderr)
        return 1
    if not prompt_text and not sys.stdin.isatty():
        prompt_text = sys.stdin.read()
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        print("[error] no prompt SYM_EMDASH pass -p \"...\" or pipe text on stdin", file=sys.stderr)
        return 1

    resolve_identity()
    register_extensions(quiet=True)
    if max_turns:
        globals()["MAX_STEPS"] = int(max_turns)
    _NONINTERACTIVE = True
    _PLAN_APPROVER = lambda plan: ("no", "")  # never block a headless run on plan approval

    CURRENT_SESSION["id"] = new_session_id()
    CURRENT_SESSION["created"] = _now()
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt_text}]

    collected = {"text": [], "final": "", "tools": [], "error": None}

    def sink(obj):
        t = obj.get("type")
        if output_format == "stream-json":
            _default_emit(obj)
        if t == "token":
            collected["text"].append(obj.get("delta", ""))
        elif t == "turn_end":
            collected["final"] = obj.get("content", "") or "".join(collected["text"])
        elif t == "tool_start":
            collected["tools"].append(obj.get("name"))
        elif t == "error":
            collected["error"] = obj.get("message")

    _EMIT_SINK = sink
    try:
        _serve_run(messages)
    except Exception as exc:  # noqa: BLE001
        collected["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _EMIT_SINK = _default_emit
        save_session(messages)
        close_extensions()

    result = collected["final"] or "".join(collected["text"])
    if output_format == "json":
        _default_emit({
            "type": "result",
            "subtype": "error" if collected["error"] else "success",
            "result": result,
            "is_error": bool(collected["error"]),
            "error": collected["error"],
            "session_id": CURRENT_SESSION["id"],
            "num_turns": SESSION.get("requests", 0),
            "tools_used": collected["tools"],
            "usage": dict(SESSION),
            "cost": _session_cost(),
        })
    elif output_format == "stream-json":
        _default_emit({"type": "result", "result": result,
                       "is_error": bool(collected["error"]),
                       "session_id": CURRENT_SESSION["id"]})
    else:  # text
        if collected["error"]:
            print(f"[error] {collected['error']}", file=sys.stderr)
        print(result)
    return 1 if collected["error"] else 0


def serve_main():
    global _INSTALL_APPROVER, _PLAN_APPROVER
    if not API_URL:
        _emit({"type": "error", "message": "no server set SYM_EMDASH set HERA_API_URL"})
        return
    resolve_identity()  # label sessions by the key's account email (fail-silent)
    # Reactive missing-binary installs and plan approval ask via the editor.
    _INSTALL_APPROVER = _serve_install_approver
    _PLAN_APPROVER = _serve_plan_approver
    register_extensions(quiet=True)
    CURRENT_SESSION["id"] = new_session_id()
    CURRENT_SESSION["created"] = _now()
    messages = [{"role": "system", "content": system_prompt()}]
    threading.Thread(target=_serve_input_thread, daemon=True).start()
    _emit({"type": "ready", "name": NAME, "model": MODEL, "cwd": os.getcwd(),
           "sandbox": sandbox_label(), "tools": list(TOOLS), "auto_mode": AUTO_MODE,
           "user": whoami_label() if (USER_EMAIL or USER_NAME) else "",
           "vision": bool(VISION_URL), "needs_key": not bool(API_KEY)})
    while True:
        msg = _MAIN_Q.get()
        if msg is None:
            break
        t = msg.get("type")
        if t == "exit":
            break
        if t == "prompt":
            _INTERRUPT.clear()  # fresh turn SYM_EMDASH drop any stale interrupt
            imgs = [(f"image-{i + 1}", du)
                    for i, du in enumerate(msg.get("images") or [])]
            content, attached = expand_mentions(msg.get("text", ""), extra_images=imgs)
            if attached:
                _emit({"type": "info", "text": f"attached: {', '.join(attached)}"})
            messages.append({"role": "user", "content": content})
            try:
                _serve_run(messages)
            except Exception as exc:  # noqa: BLE001
                _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            save_session(messages)
        elif t == "undo":
            _emit({"type": "info", "text": undo_last()})
        elif t == "clear":
            messages[:] = [{"role": "system", "content": system_prompt()}]
            _always_ok.clear()
            _emit({"type": "info", "text": "conversation cleared"})
    close_extensions()


# ── REPL ──────────────────────────────────────────────────────────────────────
def _start_new_session():
    CURRENT_SESSION["id"] = new_session_id()
    CURRENT_SESSION["created"] = _now()
    for k in SESSION:
        SESSION[k] = 0
    return [{"role": "system", "content": system_prompt()}]


def _parse_ver(s):
    """'0.4.0' -> (0, 4, 0); non-numeric parts -> 0. Safe for comparison."""
    parts = []
    for p in str(s).strip().split("."):
        m = re.match(r"\d+", p)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts) or (0,)


def _update_source():
    """Where to fetch the latest version string. Configurable; defaults to the
    public GitHub copy that both download paths are published from."""
    return _cfg("HERA_UPDATE_URL", key="update_url",
                default="https://raw.githubusercontent.com/jones0011738/hera-cli/main/VERSION")


def check_for_update():
    """Print a one-line notice if a newer version is published. Throttled to
    once/day and fail-silent SYM_EMDASH never blocks startup or errors out.

    A previously-seen newer version is remembered in the config, so the notice
    keeps showing on every launch until the user actually updates.
    """
    if _truthy(_env("HERA_NO_UPDATE_CHECK")):
        return
    now = int(time.time())
    latest = _FILE_CFG.get("latest_known_version", "")
    last   = int(_FILE_CFG.get("last_update_check", 0) or 0)

    if now - last >= 86400:  # at most once per day
        try:
            r = requests.get(_update_source(), timeout=2,
                             headers={"User-Agent": _WEB_UA})
            if r.ok:
                remote = r.text.strip().split()[0][:20]
                if remote:
                    latest = remote
            save_config({"latest_known_version": latest, "last_update_check": str(now)})
        except (requests.exceptions.RequestException, IndexError):
            save_config({"last_update_check": str(now)})  # back off even on failure

    if latest and _parse_ver(latest) > _parse_ver(VERSION):
        how = "re-run the installer, or:  curl -fsSL %s -o \"$(command -v hera || echo ~/.local/bin/hera)\"" % (
            _cfg("HERA_DOWNLOAD_URL", key="download_url",
                 default="https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py"))
        print(f"{YELL}{SYM_ARROW_U} update available: {NAME} {latest}{R} {DIM}(you have {VERSION}){R}\n"
              f"  {DIM}{how}{R}\n")


def onboard():
    """First-run setup: if the endpoint/key are missing, capture them once and
    persist to the config file so the user never has to export env vars.

    The installer pre-writes `api_url`, so in the normal flow the user only
    pastes their key. Env vars still override everything.
    """
    global API_URL, API_KEY
    if API_URL and API_KEY:
        return True
    if not sys.stdin.isatty():
        return bool(API_URL and API_KEY)  # non-interactive: let caller's guards report

    print(f"\n{ACCENT}{SYM_HALF_L}{R} {BOLD}Welcome to {NAME}{R}  {GREY}{SYM_MIDDOT} one-time setup{R}\n")

    if not API_URL:
        print(f"{DIM}Your endpoint is the identity proxy, e.g. http://<host>:8090/v1{R}")
        try:
            url = input(f"{BOLD}  Endpoint URL: {R}").strip().rstrip("/")
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not url:
            return False
        API_URL = url

    if not API_KEY:
        print(f"\n{DIM}Your personal API key from Open WebUI {SYM_ARROW_R} Settings {SYM_ARROW_R} Account {SYM_ARROW_R} API Keys.{R}")
        try:
            key = input(f"{BOLD}  Paste your API key: {R}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not key:
            return False
        API_KEY = key

    save_config({"api_url": API_URL, "api_key": API_KEY})
    print(f"\n{GREEN}{SYM_CHECK} saved to {CONFIG_PATH}{R} {DIM}{SYM_EMDASH} you're set; this won't ask again.{R}\n")
    return True


def logout():
    """Forget the saved API key + identity so a different user can sign in.

    Keeps the endpoint (so the next user only pastes their key). Removes the key,
    resolved email and name from the config file and from this process.
    """
    global API_KEY, USER_EMAIL, USER_NAME, USER_ID, SESSIONS_DIR
    for k in ("api_key", "user", "user_name"):
        _FILE_CFG.pop(k, None)
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_FILE_CFG, f, indent=2)
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass
    API_KEY = ""
    USER_EMAIL = USER_NAME = ""
    USER_ID = _compute_user_id()
    SESSIONS_DIR = _sessions_dir_for(USER_ID)


def _self_update(force=False):
    """Download the latest hera.py over the currently-running file.

    Returns (True, msg) if updated, (None, msg) if already current, (False, msg)
    on error. Source is the install server's `download_url` (recorded by the
    installer), falling back to the public GitHub copy.
    """
    download_url = _cfg("HERA_DOWNLOAD_URL", key="download_url",
                        default="https://raw.githubusercontent.com/jones0011738/hera-cli/main/hera.py")
    target = sys.argv[0] if (sys.argv and os.path.isfile(sys.argv[0])) else __file__
    target = os.path.realpath(target)
    try:
        r = requests.get(download_url, timeout=20, headers={"User-Agent": _WEB_UA})
        r.raise_for_status()
        src = r.text
    except requests.exceptions.RequestException as exc:
        return False, f"download failed from {download_url} ({exc})"
    m = re.search(r'^VERSION\s*=\s*"([^"]+)"', src, re.M)
    remote = m.group(1) if m else None
    if not remote or "def main()" not in src or len(src) < 5000:
        return False, "the downloaded file doesn't look like hera.py SYM_EMDASH aborted (nothing changed)"
    if not force and _parse_ver(remote) <= _parse_ver(VERSION):
        return None, f"already up to date (v{VERSION})"
    try:
        tmp = target + ".new"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(src)
        os.chmod(tmp, 0o755)
        os.replace(tmp, target)
    except OSError as exc:
        return False, (f"couldn't write {target} ({exc}); "
                       f"manual: curl -fsSL {download_url} -o {target}")
    save_config({"latest_known_version": remote})
    return True, f"updated v{VERSION} {SYM_ARROW_R} v{remote}  {SYM_MIDDOT}  {target}"


def doctor():
    """`hera doctor` SYM_EMDASH self-update to the latest version, then check health."""
    def line(good, label, val):
        mark = f"{GREEN}{SYM_CHECK}{R}" if good else f"{RED}{SYM_CROSS}{R}"
        print(f"  {mark} {label:<12} {DIM}{val}{R}")

    print(f"\n{ACCENT}{SYM_HALF_L}{R} {BOLD}{NAME} doctor{R}  {GREY}{SYM_MIDDOT} update + health check{R}\n")

    status, msg = _self_update(force=("--force" in sys.argv))
    line(status is not False, "update", msg)
    updated = status is True

    line(bool(API_URL), "endpoint", API_URL or "(unset SYM_EMDASH run `hera` to set it)")
    line(bool(API_KEY), "api key", "set" if API_KEY else "(unset)")

    if API_URL and API_KEY:
        try:
            r = requests.post(f"{API_URL}/chat/completions",
                              json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                                    "max_tokens": 1, "stream": False},
                              headers={"Authorization": f"Bearer {API_KEY}",
                                       "Content-Type": "application/json"}, timeout=20)
            line(r.ok, "model", f"{MODEL} {SYM_EMDASH} HTTP {r.status_code}"
                 + ("" if r.ok else f" {SYM_MIDDOT} {(r.text or '').strip()[:120]}"))
        except requests.exceptions.RequestException as exc:
            line(False, "model", f"unreachable: {exc}")
        try:
            resolve_identity()
            line(bool(USER_EMAIL or USER_NAME), "identity", whoami_label())
        except Exception:  # noqa: BLE001
            line(True, "identity", USER_ID)

    line(True, "sandbox", sandbox_label())
    line(True, "context", f"auto-compacts near {CONTEXT_TOKENS} tok, and self-recovers on overflow")
    if updated:
        print(f"\n  {GREEN}Updated {SYM_EMDASH} re-run `hera` to use the new version.{R}\n")
    else:
        print()


def main():
    global HIDE_REASONING

    ap = argparse.ArgumentParser(prog="hera", add_help=True,
                                 description="Hera SYM_EMDASH agentic coding CLI")
    ap.add_argument("command", nargs="?", default=None,
                    help="doctor SYM_MIDDOT logout SYM_MIDDOT whoami SYM_MIDDOT mcp-login <server>")
    ap.add_argument("extra", nargs="?", default=None, help="argument for `command` (e.g. server name)")
    ap.add_argument("--resume", "-r", nargs="?", const="__latest__", default=None,
                    metavar="ID", help="resume a saved session (latest if no ID)")
    ap.add_argument("--continue", "-c", dest="cont", action="store_true",
                    help="continue the most recent session")
    ap.add_argument("--list-sessions", "-l", action="store_true",
                    help="list saved sessions and exit")
    ap.add_argument("--serve", action="store_true",
                    help="headless JSON mode over stdin/stdout (used by the VS Code extension)")
    ap.add_argument("--print", "-p", dest="print_prompt", nargs="?", const="", default=None,
                    metavar="PROMPT",
                    help="headless one-shot: run PROMPT (or stdin) and print the result, then exit")
    ap.add_argument("--output-format", dest="output_format", default="text",
                    choices=["text", "json", "stream-json"],
                    help="with -p: output as text (default), json, or stream-json")
    ap.add_argument("--max-turns", dest="max_turns", type=int, default=None,
                    help="with -p: cap the number of agentic tool round-trips")
    ap.add_argument("--yolo", action="store_true",
                    help="auto-approve every tool call (also via HERA_YOLO=1)")
    ap.add_argument("--add-dir", dest="add_dir", action="append", default=[], metavar="DIR",
                    help="grant access to an extra directory beyond cwd (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="with `doctor`: re-download even if already up to date")
    ap.add_argument("--version", "-V", action="version", version=f"{NAME} {VERSION}")
    args = ap.parse_args()

    if args.yolo:
        globals()["YOLO"] = True
    for d in args.add_dir:
        add_extra_dir(d)

    # Headless one-shot print mode (Claude-Code `-p`). Honors stdin when given
    # with no value: `echo "fix the bug" | hera -p`.
    if args.print_prompt is not None:
        sys.exit(print_main(args.print_prompt, args.output_format, args.max_turns))

    if args.command == "doctor":
        doctor()
        return
    if args.command == "logout":
        logout()
        print(f"{GREEN}{SYM_CHECK} logged out{R} {DIM}{SYM_EMDASH} key and identity cleared from {CONFIG_PATH}. "
              f"Run `hera` to sign in with a different key.{R}")
        return
    if args.command == "whoami":
        resolve_identity(force=True)
        print(whoami_label())
        return
    if args.command == "mcp-login":
        print(mcp_oauth_login(args.extra))
        return
    if args.command:
        print(f"{RED}[error] unknown command {args.command!r}. "
              f"Try: hera doctor {SYM_MIDDOT} hera logout {SYM_MIDDOT} hera whoami {SYM_MIDDOT} hera mcp-login <server>{R}",
              file=sys.stderr)
        return

    if args.serve:
        serve_main()
        return

    if args.list_sessions:
        resolve_identity()
        print_sessions()
        return

    onboard()
    resolve_identity()  # key SYM_ARROW_R account email, so sessions are labelled by who you are
    check_for_update()  # one-line notice if a newer version is published (fail-silent)
    if not API_URL:
        print(f"{RED}[error] no endpoint set. Run `hera` interactively to set it, or:\n"
              f"  export HERA_API_URL=http://<host>:8090/v1   # the identity proxy\n"
              f"  export HERA_API_KEY=<your personal key>{R}", file=sys.stderr)
        return
    if not API_KEY:
        print(f"{YELL}[warn] no API key set {SYM_EMDASH} the server will reject requests with 401.\n"
              f"       run `hera` interactively to paste your key, or export HERA_API_KEY.{R}\n",
              file=sys.stderr)

    register_extensions()
    setup_completion()  # Tab-completion + recommendations for slash commands

    # Resume or start fresh.
    resume_target = "__latest__" if args.cont else args.resume
    messages = None
    if resume_target is not None:
        s = load_session(resume_target)
        if s:
            messages = s["messages"]
            CURRENT_SESSION["id"] = s["id"]
            CURRENT_SESSION["created"] = s.get("created")
            CURRENT_SESSION["title"] = s.get("title") or _first_user(s.get("messages") or [])
            SESSION.update(s.get("tokens", {}))
            print(f"{DIM}resumed: {_session_label(s)} "
                  f"({sum(1 for m in messages if m.get('role') == 'user')} messages, "
                  f"{SESSION.get('total', 0)} tokens){R}")
        else:
            print(f"{YELL}no matching session {SYM_EMDASH} starting fresh{R}")
    if messages is None:
        messages = _start_new_session()

    print_banner()
    spinner = Spinner()

    # SessionStart hooks may print a status line and inject startup context.
    _run_hooks("SessionStart")
    if _HOOK_CONTEXT and messages:
        messages[0]["content"] += f"\n\n<session-start-context>\n{_HOOK_CONTEXT}\n</session-start-context>"
    _emit_telemetry("session_start")
    _emit_metric("hera.sessions", 1)

    try:
        _repl(messages, spinner)
    finally:
        _run_hooks("SessionEnd")
        _emit_telemetry("session_end", total_tokens=SESSION.get("total", 0),
                        requests=SESSION.get("requests", 0))
        save_session(messages)
        close_extensions()


def _builtin_statusline():
    """Built-in status line SYM_EMDASH model SYM_MIDDOT git branch SYM_MIDDOT tokens SYM_MIDDOT active modes."""
    parts = [f"{BOLD}{MODEL.split('/')[-1]}{R}"]
    try:
        r = subprocess.run("git branch --show-current", shell=True,
                           capture_output=True, text=True, cwd=os.getcwd(), timeout=1)
        if r.returncode == 0 and r.stdout.strip():
            parts.append(f"git:{r.stdout.strip()}")
    except Exception:
        pass
    t = SESSION.get("total", 0)
    if t:
        parts.append(f"{t:,} tok")
    if PLAN_MODE:
        parts.append(f"{YELL}plan{R}")
    if AUTO_MODE != "read":
        parts.append(f"auto:{AUTO_MODE}")
    if THINK_LEVEL not in ("normal", ""):
        parts.append(f"think:{THINK_LEVEL}")
    print(f"{DIM}{'  {SYM_MIDDOT}  '.join(parts)}{R}")


def _render_statusline():
    """Run the configured statusline command (session JSON on stdin) and print
    its stdout above the prompt, like Claude Code's statusLine."""
    if not STATUSLINE_CMD:
        return
    if STATUSLINE_CMD == "builtin":
        _builtin_statusline()
        return
    payload = json.dumps({"model": MODEL, "cwd": os.getcwd(),
                          "session_tokens": SESSION.get("total", 0),
                          "auto_mode": AUTO_MODE, "plan_mode": PLAN_MODE,
                          "output_style": OUTPUT_STYLE, "think": THINK_LEVEL})
    try:
        r = subprocess.run(STATUSLINE_CMD, shell=True, input=payload, text=True,
                           capture_output=True, timeout=5, cwd=os.getcwd())
        line = (r.stdout or "").strip()
        if line:
            print(f"{DIM}{line}{R}")
    except Exception:  # noqa: BLE001 SYM_EMDASH a broken statusline must not break the REPL
        pass


def _repl(messages, spinner):
    global HIDE_REASONING
    while True:
        _render_statusline()
        try:
            user_input = read_line(f"{ACCENT}{BOLD}{SYM_PROMPT}{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Session ended.{R}")
            break

        if not user_input:
            continue

        # `# fact` quick-add to memory (Claude-Code style). `# user fact` → user
        # scope (all projects); otherwise project scope (./HERA.md).
        if user_input.startswith("#"):
            body = user_input[1:].strip()
            scope = "project"
            if body.lower().startswith("user "):
                scope, body = "user", body[5:].strip()
            print(f"\n{DIM}{add_memory(body, scope)}{R}\n")
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit"):
            print(f"{DIM}Session ended.{R}")
            break
        if cmd in ("/clear", "/new"):
            save_session(messages)
            messages[:] = _start_new_session()
            _always_ok.clear()
            reset_turn_marks()
            print(f"\n{DIM}Started a fresh session ({CURRENT_SESSION['id']}).{R}\n")
            continue
        if cmd == "/logout":
            save_session(messages)                 # keep the current user's history
            logout()
            print(f"\n{GREEN}{SYM_CHECK} logged out{R} {DIM}{SYM_EMDASH} key cleared. Sign in with a different key.{R}")
            if not onboard():                      # prompts for the new key (endpoint kept)
                print(f"{DIM}No key entered {SYM_EMDASH} exiting.{R}")
                break
            resolve_identity()
            messages[:] = _start_new_session()     # don't show the previous user's context
            _always_ok.clear()
            reset_turn_marks()
            print(f"{DIM}Fresh session started for the new account.{R}\n")
            continue
        if cmd == "/sessions":
            print_sessions(project_only=True)
            continue
        if cmd in ("/resume", "/history"):
            resume_picker(messages)
            continue
        if cmd.startswith("/resume "):
            sid = user_input[8:].strip()
            s = load_session(sid)
            if s:
                _switch_to(messages, s)
            else:
                print(f"\n{DIM}no session matching {sid!r}. try /resume to pick from a list.{R}\n")
            continue
        if cmd == "/undo":
            print(f"\n{DIM}{undo_last()}{R}\n")
            continue
        if cmd == "/rewind" or cmd.startswith("/rewind "):
            arg = user_input[7:].strip()
            rewind_picker(messages, int(arg) if arg.isdigit() else None)
            continue
        if cmd == "/context":
            context_report(messages)
            continue
        if cmd == "/memory":
            memory_report()
            continue
        if cmd == "/export" or cmd.startswith("/export "):
            print(f"\n{DIM}{export_conversation(messages, user_input[7:].strip() or None)}{R}\n")
            continue
        if cmd == "/init":
            mark_turn(messages, "/init")
            messages.append({"role": "user", "content": _init_prompt()})
            try:
                ok = run_agent(messages, spinner)
            except KeyboardInterrupt:
                spinner.stop(); print(f"\n{DIM}(interrupted){R}\n"); ok = True
            if not ok:
                del messages[len(messages) - 1:]
                if TURN_MARKS:
                    TURN_MARKS.pop()
            save_session(messages)
            continue
        if cmd == "/plugins" or cmd.startswith("/plugins "):
            arg = user_input[8:].strip()
            if arg.startswith("install "):
                msg = install_plugin(arg[8:].strip())
                print(f"\n{DIM}{msg}{R}")
                if not msg.startswith("[error]"):
                    register_extensions(quiet=True)  # pick up the new plugin now
                print()
            elif arg in ("market", "marketplace", "available"):
                cat = marketplace_catalog()
                if not cat:
                    print(f"\n{DIM}no marketplaces. Add index files in {MARKETPLACES_DIR}/<name>.json{R}\n")
                else:
                    print(f"\n{BOLD}Available plugins{R}")
                    for p in cat:
                        print(f"  {CYAN}{p['name']}{R} {DIM}[{p['marketplace']}] "
                              f"{p.get('description', '')}{R}")
                    print(f"{DIM}  install with {R}{CYAN}/plugins install <name>{R}\n")
            else:
                if not PLUGINS:
                    print(f"\n{DIM}no plugins installed. See {R}{CYAN}/plugins marketplace{R}"
                          f"{DIM}; install dirs live in {PLUGINS_DIR}{R}\n")
                else:
                    print(f"\n{BOLD}Installed plugins{R}")
                    for p in PLUGINS:
                        bits = []
                        if p["commands"]:
                            bits.append(f"{len(p['commands'])} cmd")
                        if p["agents"]:
                            bits.append(f"{len(p['agents'])} agent")
                        if p["mcp"]:
                            bits.append("mcp")
                        print(f"  {CYAN}{p['name']}{R} {DIM}{p['version']} "
                              f"{'{SYM_MIDDOT} '.join(bits)} {SYM_EMDASH} {p['description']}{R}")
                    print()
            continue
        if cmd == "/doctor":
            _run_doctor()
            continue
        if cmd == "/review":
            inrepo = subprocess.run("git rev-parse --is-inside-work-tree",
                                    shell=True, capture_output=True, text=True, cwd=os.getcwd())
            if inrepo.returncode != 0:
                print(f"\n{DIM}not a git repository {SYM_EMDASH} /review needs git{R}\n")
                continue
            diff = subprocess.run("git diff HEAD", shell=True,
                                  capture_output=True, text=True, cwd=os.getcwd()).stdout
            if not diff.strip():
                diff = subprocess.run("git diff --cached", shell=True,
                                      capture_output=True, text=True, cwd=os.getcwd()).stdout
            if not diff.strip():
                print(f"\n{DIM}no changes to review (working tree is clean){R}\n")
                continue
            mark_turn(messages, "/review")
            messages.append({"role": "user", "content":
                f"Please review this git diff for bugs, issues, style problems, and improvements. "
                f"Be specific about file and line. Group findings by severity (critical / warning / suggestion).\n\n"
                f"```diff\n{diff[:10000]}\n```"})
            try:
                ok = run_agent(messages, spinner)
            except KeyboardInterrupt:
                spinner.stop(); print(f"\n{DIM}(interrupted){R}\n"); ok = True
            if not ok:
                del messages[len(messages) - 1:]
                if TURN_MARKS: TURN_MARKS.pop()
            save_session(messages)
            continue
        if cmd == "/pr" or cmd.startswith("/pr "):
            title_hint = user_input[3:].strip()
            mark_turn(messages, "/pr")
            pr_prompt = (
                "Create a GitHub pull request for the current branch. "
                "Steps: (1) run `git log main..HEAD --oneline` to summarise what changed, "
                "(2) run `git branch --show-current` to confirm the branch name, "
                "(3) use `gh pr create` with a clear title and a brief body (## Summary bullet points + ## Test plan). "
            )
            if title_hint:
                pr_prompt += f"Use this as the PR title: {title_hint!r}. "
            pr_prompt += (
                "If `gh` is not installed, call tool_install('gh', 'GitHub CLI for PR creation') first. "
                "If `gh auth status` fails, tell the user to run `gh auth login` in a separate terminal."
            )
            messages.append({"role": "user", "content": pr_prompt})
            try:
                ok = run_agent(messages, spinner)
            except KeyboardInterrupt:
                spinner.stop(); print(f"\n{DIM}(interrupted){R}\n"); ok = True
            if not ok:
                del messages[len(messages) - 1:]
                if TURN_MARKS: TURN_MARKS.pop()
            save_session(messages)
            continue
        if cmd == "/diff":
            inrepo = subprocess.run("git rev-parse --is-inside-work-tree",
                                    shell=True, capture_output=True, text=True, cwd=os.getcwd())
            if inrepo.returncode != 0:
                print(f"\n{DIM}not a git repository {SYM_EMDASH} /diff needs git{R}\n")
                continue
            proc = subprocess.run("git diff --stat && echo '---' && git diff",
                                  shell=True, capture_output=True, text=True, cwd=os.getcwd())
            out = (proc.stdout or "").rstrip() or "(no changes)"
            print(f"\n{DIM}{out[:6000]}{R}\n")
            continue
        if cmd == "/compact":
            print(f"\n{DIM}compacting{SYM_ELLIPSIS}{R}")
            print(f"{DIM}{compact_history(messages)}{R}\n")
            save_session(messages)
            continue
        if cmd == "/help":
            print_help()
            continue
        if cmd == "/skills":
            print_shared_skills()
            continue
        if cmd.startswith("/skills "):
            print_shared_skills(user_input[8:].strip())
            continue
        if cmd == "/tokens":
            s = SESSION
            print(f"\n{DIM}session: {s['total']} tokens over {s['requests']} requests "
                  f"(prompt {s['prompt']} / completion {s['completion']}){_cost_suffix()}{R}\n")
            continue
        if cmd == "/tools":
            print(f"\n{DIM}tools: {', '.join(TOOLS)}\n"
                  f"  approval required: {', '.join(sorted(SIDE_EFFECTS))}\n"
                  f"  run_bash sandbox: {sandbox_label()}{R}\n")
            continue
        if cmd == "/sandbox":
            print(f"\n{DIM}sandbox: {sandbox_label()}\n"
                  f"  mode={SANDBOX_MODE} kind={SANDBOX_KIND} network={'on' if SANDBOX_NET else 'off'}\n"
                  f"  change with HERA_SANDBOX=bwrap|unshare|none and HERA_SANDBOX_NET=0|1{R}\n")
            continue
        if cmd == "/allow" or cmd.startswith("/allow "):
            arg = user_input[6:].strip()
            if arg:
                ALLOW_PATTERNS.append(arg)
                print(f"\n{DIM}added allow pattern: {arg!r}  ({len(ALLOW_PATTERNS)} total){R}\n")
            elif ALLOW_PATTERNS:
                print(f"\n{DIM}run_bash allow patterns ({len(ALLOW_PATTERNS)}):\n  "
                      + "\n  ".join(ALLOW_PATTERNS) + f"{R}\n")
            else:
                print(f"\n{DIM}no allow patterns. Add one with: /allow <pattern>  "
                      f"(e.g. /allow git status){R}\n")
            continue
        if cmd == "/cwd":
            print(f"\n{DIM}cwd: {os.getcwd()}{R}\n")
            continue
        if cmd == "/agents":
            if CUSTOM_AGENTS:
                print(f"\n{BOLD}Named sub-agents{R} {DIM}(call via the task tool's agent field){R}")
                for name, body in CUSTOM_AGENTS.items():
                    meta, _ = _parse_frontmatter(body)
                    tools = meta.get("tools", "all tools")
                    print(f"  {CYAN}{name}{R} {DIM}{SYM_EMDASH} {tools}{R}")
                print()
            else:
                print(f"\n{DIM}no named agents. Add markdown files in {AGENTS_DIR}/<name>.md "
                      f"(optional 'tools:' frontmatter).{R}\n")
            continue
        if cmd == "/mcp" or cmd.startswith("/mcp "):
            arg = user_input[4:].strip()
            if arg.startswith("login"):
                print(f"\n{DIM}{mcp_oauth_login(arg[5:].strip())}{R}\n")
            elif _mcp_clients:
                print(f"\n{BOLD}MCP servers{R}")
                for c in _mcp_clients:
                    print(f"  {CYAN}{c.name}{R} {DIM}{SYM_EMDASH} {len(c.tools)} tool(s){R}")
                print(f"{DIM}  authenticate: {R}{CYAN}/mcp login <server>{R}\n")
            else:
                print(f"\n{DIM}no MCP servers connected. Configure them in {MCP_CONFIG} "
                      f"(or bundle mcp.json in a plugin). OAuth: {R}{CYAN}/mcp login <server>{R}\n")
            continue
        if cmd == "/permissions" or cmd.startswith("/permissions "):
            parts = user_input.split(None, 2)
            if len(parts) >= 3:
                print(f"\n{DIM}{add_permission(parts[1].lower(), parts[2])}{R}\n")
            else:
                print(f"\n{BOLD}Permissions{R} {DIM}(managed rules win; then these){R}")
                for label, perms in (("managed", _MANAGED_PERMS), ("user", _PERMS)):
                    for bucket in ("deny", "ask", "allow"):
                        for rule in (perms.get(bucket) or []):
                            print(f"  {DIM}{label:8}{R}{CYAN}{bucket:5}{R} {rule}")
                print(f"{DIM}  add: {R}{CYAN}/permissions deny run_bash(rm *){R}{DIM} {SYM_MIDDOT} "
                      f"allow {SYM_MIDDOT} ask{R}\n")
            continue
        if cmd == "/config":
            print(f"\n{BOLD}Config{R}")
            for k, v in _config_summary().items():
                print(f"  {DIM}{k:13}{R}{v}")
            print()
            continue
        if cmd == "/add-dir" or cmd.startswith("/add-dir "):
            arg = user_input[8:].strip()
            if arg:
                print(f"\n{DIM}{add_extra_dir(arg)}{R}\n")
            elif EXTRA_DIRS:
                print(f"\n{DIM}trusted dirs (beyond cwd):\n  " + "\n  ".join(EXTRA_DIRS) + f"{R}\n")
            else:
                print(f"\n{DIM}no extra dirs. Add one with {R}{CYAN}/add-dir <path>{R}{DIM}.{R}\n")
            continue
        if cmd == "/reasoning":
            HIDE_REASONING = not HIDE_REASONING
            state = "hidden" if HIDE_REASONING else "visible"
            print(f"\n{DIM}reasoning is now {state}.{R}\n")
            continue
        if cmd == "/vim":
            globals()["VIM_MODE"] = not VIM_MODE
            print(f"\n{DIM}vim keybindings {'ON' if VIM_MODE else 'OFF'} {SYM_EMDASH} each prompt starts in "
                  f"INSERT; press {R}{CYAN}Esc{R}{DIM} for NORMAL (h l 0 $ w b e i a A I x D dd dw cw).{R}\n")
            continue
        if cmd == "/think" or cmd.startswith("/think "):
            arg = user_input[6:].strip().lower()
            if arg in ("off", "normal", "hard", "max"):
                globals()["THINK_LEVEL"] = arg
                save_config({"think": arg})
                blurb = {"off": "no model thinking (fastest)",
                         "normal": "default thinking",
                         "hard": "deeper thinking enabled",
                         "max": "deepest thinking enabled"}[arg]
                print(f"\n{DIM}thinking {SYM_ARROW_R} {BOLD}{arg}{R}{DIM} ({blurb}).{R}\n")
            else:
                print(f"\n{DIM}thinking is {BOLD}{THINK_LEVEL}{R}{DIM}. "
                      f"Use {R}{CYAN}/think off|normal|hard|max{R}{DIM} "
                      f"(or say 'ultrathink' in a prompt).{R}\n")
            continue
        if cmd == "/output-style" or cmd.startswith("/output-style "):
            arg = user_input[14:].strip()
            avail = output_styles_available()
            if arg in avail:
                globals()["OUTPUT_STYLE"] = arg
                save_config({"output_style": arg})
                print(f"\n{DIM}output style {SYM_ARROW_R} {BOLD}{arg}{R}{DIM}.{R}\n")
            elif not arg:
                print(f"\n{DIM}output style is {BOLD}{OUTPUT_STYLE}{R}{DIM}. available: "
                      f"{', '.join(avail)}.\n  add custom ones in "
                      f"{os.path.join(CONFIG_DIR, 'output-styles')}/<name>.md{R}\n")
            else:
                print(f"\n{DIM}unknown style {arg!r}. available: {', '.join(avail)}.{R}\n")
            continue
        if cmd == "/statusline":
            print(f"\n{DIM}statusline: {STATUSLINE_CMD or '(none)'}\n"
                  f"  set with HERA_STATUSLINE='<cmd>' or config \"statusline\" "
                  f"(session JSON on stdin {SYM_ARROW_R} stdout shown above the prompt).{R}\n")
            continue
        if cmd == "/plan":
            globals()["PLAN_MODE"] = not PLAN_MODE
            if PLAN_MODE:
                print(f"\n{DIM}plan mode ON {SYM_EMDASH} I'll research read-only and present a plan for you to "
                      f"approve before making any changes. Toggle off with /plan.{R}\n")
            else:
                print(f"\n{DIM}plan mode OFF {SYM_EMDASH} I can edit and run again.{R}\n")
            continue
        if cmd == "/todos":
            if TODOS:
                print("\n" + _render_todos_text() + "\n")
            else:
                print(f"\n{DIM}no to-do list yet.{R}\n")
            continue
        if cmd == "/auto" or cmd.startswith("/auto "):
            arg = user_input[5:].strip().lower()
            alias = {"off": "read", "none": "read", "write": "edit", "edits": "edit",
                     "readwrite": "edit", "read+write": "edit", "everything": "all", "yolo": "all"}
            mode = alias.get(arg, arg)
            if mode in _AUTO_LEVELS:
                capped = _managed_cap_auto_mode(mode)
                if capped != mode:
                    print(f"\n{YELL}managed policy caps auto mode at {capped}.{R}")
                mode = capped
                globals()["AUTO_MODE"] = mode
                _save_auto_mode(mode)
                blurb = {"read": "auto-approve reads only SYM_EMDASH writes/commands will prompt",
                         "edit": "auto-approve reads + file edits SYM_EMDASH shell commands still prompt",
                         "all":  "auto-approve ALL tools (deny rules & plan mode still apply)"}[mode]
                print(f"\n{DIM}auto mode {SYM_ARROW_R} {BOLD}{mode}{R}{DIM} for this project ({_project_key()}).\n"
                      f"  {blurb}.  Stop any time with {R}{CYAN}/auto off{R}{DIM}.{R}\n")
            elif not arg:
                print(f"\n{DIM}auto mode is {BOLD}{AUTO_MODE}{R}{DIM} for this project.\n"
                      f"  {R}{CYAN}/auto read{R}{DIM} (reads only) {SYM_MIDDOT} {R}{CYAN}/auto edit{R}{DIM} "
                      f"(+ file edits) {SYM_MIDDOT} {R}{CYAN}/auto all{R}{DIM} (everything) {SYM_MIDDOT} "
                      f"{R}{CYAN}/auto off{R}{DIM} (stop).{R}\n")
            else:
                print(f"\n{DIM}unknown auto mode {arg!r}. Use read / edit / all / off.{R}\n")
            continue

        # User-defined slash commands from ~/.config/hera/commands/*.md.
        first = user_input.split()[0]
        cust = CUSTOM_COMMANDS.get(first[1:]) if first.startswith("/") else None
        if cust is not None:
            _, body = _parse_frontmatter(cust)
            cmd_args = user_input[len(first):].strip()
            prompt_text = body.replace("$ARGUMENTS", cmd_args).strip()
            mark = len(messages)
            mark_turn(messages, user_input)
            messages.append({"role": "user", "content": prompt_text})
            try:
                ok = run_agent(messages, spinner)
            except KeyboardInterrupt:
                spinner.stop(); print(f"\n{DIM}(interrupted){R}\n"); ok = True
            if not ok:
                del messages[mark:]
                if TURN_MARKS:
                    TURN_MARKS.pop()
            save_session(messages)
            continue

        # Anything else that looks like a "/command" (and not a path like
        # /etc/hosts) is an unknown command — show the recommendation menu
        # instead of forwarding it to the model.
        if re.fullmatch(r"/[A-Za-z][A-Za-z-]*", first):
            print_slash_menu(user_input)
            continue

        # UserPromptSubmit hooks: may veto the prompt or inject extra context.
        blocked = _run_hooks("UserPromptSubmit", prompt=user_input)
        if blocked:
            print(f"\n{YELL}{blocked}{R}\n")
            continue
        content, attached = expand_mentions(user_input)
        if attached:
            print(f"{DIM}  {SYM_HOOKED} attached: {', '.join(attached)}{R}")
        if _HOOK_CONTEXT:  # hook stdout becomes additional context for this turn
            content = _inject_context(content, _HOOK_CONTEXT)
        mark = len(messages)  # remember where this turn starts
        mark_turn(messages, user_input)  # rewind checkpoint (conversation + files)
        messages.append({"role": "user", "content": content})
        try:
            ok = run_agent(messages, spinner)
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n{DIM}(interrupted){R}\n")
            ok = True  # keep history; user can continue
        if not ok:
            # Roll back the ENTIRE failed turn (user msg + any partial
            # assistant/tool msgs), so a transport error can't leave the
            # history in a state that breaks every subsequent request.
            del messages[mark:]
            if TURN_MARKS:
                TURN_MARKS.pop()
        save_session(messages)  # autosave after every turn


if __name__ == "__main__":
    main()
