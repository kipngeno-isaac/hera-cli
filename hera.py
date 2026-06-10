#!/usr/bin/env python3
"""
Hera — an agentic coding CLI for the Qwen3.6-35B-A3B model.

Hera runs the model in a reason→act loop with real tools: it can list
directories, find files, search code, read/write/edit files, and run shell
commands in the directory you launch it from. It asks for approval before
editing files or running commands, streams the model's reasoning, and tracks
token usage.

Required:       HERA_API_URL  (the endpoint, e.g. http://<host>:8080/v1 — no host is
                              baked in, so this file can live in a public repo)
                HERA_API_KEY  (bearer key the server enforces)
Optional:       HERA_MODEL          (default qwen3.6-35b-a3b)
                HERA_NAME           (assistant display name; default Hera)
                HERA_YOLO=1         auto-approve every tool call (no prompts)
                HERA_MAX_STEPS      max tool round-trips per message (default 25)
                HERA_HIDE_REASONING=1   don't stream the model's thinking

Legacy QWEN_* variables (and LLAMA_API_KEY) are still honoured as fallbacks.
"""
import argparse
import ast
import fnmatch
import glob as globmod
import hashlib
import importlib.util
import json
import math
import os
import re
import select
import shutil
import subprocess
import sys
import time
import threading

try:
    import readline  # noqa: F401  (enables line editing in input())
except ImportError:
    pass

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


NAME    = _env("HERA_NAME", default="Hera")
# No server host is baked in (so this repo can be public, revealing neither key
# nor host). Each user supplies HERA_API_URL + HERA_API_KEY (given on approval).
API_URL = _env("HERA_API_URL", "QWEN_API_URL", default="").rstrip("/")
MODEL   = _env("HERA_MODEL",   "QWEN_MODEL",   default="qwen3.6-35b-a3b")
API_KEY = _env("HERA_API_KEY", "QWEN_API_KEY", "LLAMA_API_KEY", default="")
YOLO    = _truthy(_env("HERA_YOLO", "QWEN_YOLO"))
MAX_STEPS      = int(_env("HERA_MAX_STEPS", "QWEN_MAX_STEPS", default="25"))
HIDE_REASONING = _truthy(_env("HERA_HIDE_REASONING", "QWEN_HIDE_REASONING"))

MAX_TOOL_OUTPUT = 12000    # chars; tool results longer than this are truncated
MAX_READ_BYTES  = 256_000  # cap read_file size

# Sandboxing for run_bash. Modes: auto | bwrap | unshare | none
SANDBOX_MODE = _env("HERA_SANDBOX", default="auto").lower()
SANDBOX_NET  = _truthy(_env("HERA_SANDBOX_NET"))  # allow network inside the sandbox

# Running token usage for the whole session.
SESSION = {"prompt": 0, "completion": 0, "total": 0, "requests": 0}


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


def sandbox_label():
    net = "network on" if SANDBOX_NET else "no network"
    if SANDBOX_KIND == "bwrap":
        return f"bwrap — fs confined to cwd, {net}"
    if SANDBOX_KIND == "unshare":
        return f"unshare — pid-isolated, {net} (install bubblewrap for fs confinement)"
    return "none — run_bash runs unconfined"


def _sandbox_argv(command):
    """Return (argv, use_shell) to execute `command` under the active sandbox."""
    cwd = os.getcwd()
    if SANDBOX_KIND == "bwrap":
        # Order matters (later mounts win): make everything read-only, give a
        # private /tmp, then bind the working dir writable LAST so it stays
        # writable even when cwd is itself under /tmp.
        argv = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--bind", cwd, cwd,
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
    return command, True  # none → shell string


def _sandbox_wrap_argv(argv):
    """Wrap a program's argv under the active sandbox (for long-running children
    like MCP servers). Returns argv unchanged when sandboxing is off."""
    cwd = os.getcwd()
    if SANDBOX_KIND == "bwrap":
        pre = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--bind", cwd, cwd, "--die-with-parent", "--chdir", cwd]
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
    "*sudo *", "*curl*|*sh*", "*wget*|*sh*",
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
TEAL  = _sgr("38;5;44")
SKY   = _sgr("38;5;39")
IND    = _sgr("38;5;33")
ACCENT = CYAN


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
                return f"  {GREY}┌─ {lang or 'code'} {'─' * max(0, 40 - len(lang))}{R}"
            return f"  {GREY}└{'─' * 46}{R}"
        if state["code"]:
            return f"  {GREY}│{R} {line}"
        if re.match(r"^#{1,6}\s", stripped):
            return f"{BOLD}{CYAN}{re.sub(r'^#{1,6}s*', '', stripped).lstrip('# ')}{R}"
        m = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if m:
            return f"{m.group(1)}  {CYAN}•{R} {_md_inline(m.group(2))}"
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)", line)
        if m:
            return f"{m.group(1)}  {CYAN}{m.group(2)}.{R} {_md_inline(m.group(3))}"
        if stripped.startswith(">"):
            return f"  {GREY}▏{R} {DIM}{stripped[1:].strip()}{R}"
        return _md_inline(line)
    except Exception:  # noqa: BLE001 — never let rendering break the stream
        return line


# ── Spinner ───────────────────────────────────────────────────────────────────
class Spinner:
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self):
        self._stop = threading.Event()
        self._t    = None
        self._t0   = None

    def _run(self, label):
        i = 0
        while not self._stop.is_set():
            secs = time.time() - self._t0
            f = self._FRAMES[i % len(self._FRAMES)]
            print(f"\r  {CYAN}{f}{R}  {DIM}{label} {secs:.1f}s{R}   ", end="", flush=True)
            i += 1
            time.sleep(0.1)

    def start(self, label="thinking…"):
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
        out += f"\n…[{len(matches) - 200} more]"
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
                            results.append(f"…[stopped at {max_results} matches]")
                            return "\n".join(results)
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable — skip
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
        out += f"\n…[{len(results) - 300} more]"
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
        for j in range(0, len(texts), 32):  # small batches → stay under embed ctx
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
        return (f"[error] old_string is not unique ({count} matches). "
                f"Add more context to make it unique, or set replace_all=true.")
    new_text = text.replace(old_string, new_string)
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_text)
    return f"Edited {p} ({count} replacement{'s' if count != 1 else ''})"


def tool_run_bash(command, timeout=120):
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
    parts = []
    if out:
        parts.append(out.rstrip("\n"))
    if err:
        parts.append(f"[stderr]\n{err.rstrip(chr(10))}")
    parts.append(f"[exit code {proc.returncode}]")
    return "\n".join(parts)


TOOLS = {
    "list_dir":   tool_list_dir,
    "read_file":  tool_read_file,
    "glob":       tool_glob,
    "search":     tool_search,
    "symbols":    tool_symbols,
    "write_file": tool_write_file,
    "edit_file":  tool_edit_file,
    "run_bash":   tool_run_bash,
}

# Tools that change the world → require approval (unless YOLO).
SIDE_EFFECTS = {"write_file", "edit_file", "run_bash"}

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
        "name": "run_bash",
        "description": ("Run a shell command in the current working directory and return "
                        "stdout, stderr, and exit code."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
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
        lines.append(f"    {RED}- …{R}")
    for ln in nlines[:8]:
        lines.append(f"    {GREEN}+ {ln[:80]}{R}")
    if len(nlines) > 8:
        lines.append(f"    {GREEN}+ …{R}")
    return "\n".join(lines)


def _preview_call(name, args):
    if name == "run_bash":
        return f"$ {args.get('command', '')}"
    if name == "write_file":
        c = args.get("content", "")
        n = c.count("\n") + 1
        return f"write {args.get('path', '?')}  ({n} lines)"
    if name == "edit_file":
        return (f"edit {args.get('path', '?')}\n"
                f"{_diff_preview(args.get('old_string', ''), args.get('new_string', ''))}")
    return f"{name}({json.dumps(args)})"


def approve(name, args):
    """Return True to run the tool, or a string denial reason."""
    if YOLO or name not in SIDE_EFFECTS or name in _always_ok:
        return True

    # run_bash: consult the command allowlist first.
    if name == "run_bash":
        cmd = args.get("command", "")
        if bash_allowed(cmd):
            print(f"  {DIM}↳ auto-approved (allowlist){R}")
            return True
        denied = _matches(cmd, DENY_PATTERNS)
        print(f"\n{YELL}{BOLD}⚠ approval needed{R} {DIM}(run_bash{', matches deny-pattern' if denied else ''}){R}")
        print(f"  $ {cmd}")
        print(f"  {DIM}sandbox: {sandbox_label()}{R}")
        try:
            ans = input(f"{BOLD}  [y]es once / [a]lways this command / "
                        f"[p]rogram (all '{cmd.split()[0] if cmd.split() else '?'}') / [n]o:{R} ").strip().lower()
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
        return "user declined to run this tool"

    # write_file / edit_file: tool-level approval.
    print(f"\n{YELL}{BOLD}⚠ approval needed{R} {DIM}({name}){R}")
    for ln in _preview_call(name, args).split("\n"):
        print(f"  {ln}")
    try:
        ans = input(f"{BOLD}  run this? [y]es / [a]lways / [n]o:{R} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "user aborted (no input)"
    if ans in ("y", "yes", ""):
        return True
    if ans in ("a", "always"):
        _always_ok.add(name)
        return True
    return "user declined to run this tool"


# ── Streaming chat call ───────────────────────────────────────────────────────
def stream_turn(messages, spinner, tools=None):
    """One model turn. Streams reasoning + content live; assembles tool calls.

    Returns dict {content, finish_reason, tool_calls, usage} or None on error.
    """
    try:
        resp = requests.post(
            f"{API_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "tools": tools if tools is not None else TOOL_SCHEMAS,
                "tool_choice": "auto",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        spinner.stop()
        print(f"{RED}[error] Cannot reach {API_URL}{R}\n", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as exc:
        spinner.stop()
        print(f"{RED}[error] {exc}{R}\n", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        spinner.stop()
        print(f"{RED}[error] Request timed out{R}\n", file=sys.stderr)
        return None

    content       = []
    tool_calls    = {}     # index -> {id, name, arguments}
    finish_reason = None
    usage         = None
    started       = False
    in_reasoning  = False

    line_buf = ""               # buffers content until a newline, then renders it
    md_state = {"code": False}

    def ensure_header():
        nonlocal started
        if not started:
            elapsed = spinner.stop()
            print(f"\n{ACCENT}▌{R} {BOLD}{NAME}{R}  {GREY}· {elapsed:.1f}s to first token{R}\n")
            started = True

    for raw in resp.iter_lines():
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

        if reasoning and not HIDE_REASONING:
            ensure_header()
            if not in_reasoning:
                print(f"{DIM}{ITAL}✶ thinking…{R}\n{DIM}", end="", flush=True)
                in_reasoning = True
            print(f"{DIM}{reasoning}{R}", end="", flush=True)

        if token:
            ensure_header()
            if in_reasoning:
                print(f"{R}\n\n", end="", flush=True)
                in_reasoning = False
            content.append(token)
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

    if in_reasoning:
        print(f"{R}\n", end="", flush=True)
    if line_buf:  # flush any trailing partial line
        print(render_md_line(line_buf, md_state), flush=True)

    return {
        "content": "".join(content),
        "finish_reason": finish_reason,
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "usage": usage,
    }


def _account(usage):
    """Fold a request's usage into the session totals; return per-request total."""
    if not usage:
        return 0
    p = usage.get("prompt_tokens", 0)
    c = usage.get("completion_tokens", 0)
    t = usage.get("total_tokens", p + c)
    SESSION["prompt"]     += p
    SESSION["completion"] += c
    SESSION["total"]      += t
    SESSION["requests"]   += 1
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


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(messages, spinner):
    """Drive the reason→act loop until the model produces a final answer."""
    turn_tokens = 0
    for step in range(MAX_STEPS):
        spinner.start()
        result = stream_turn(messages, spinner)
        if result is None:
            return False  # transport error; caller rolls back the user message

        turn_tokens += _account(result.get("usage"))
        calls = result["tool_calls"]

        assistant_msg = {"role": "assistant", "content": result["content"] or ""}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ]
        messages.append(assistant_msg)

        if not calls:
            print(f"\n{GREY}{'─' * 50}{R}")
            print(f"{GREY}  {turn_tokens} tok this turn · {SESSION['total']} session{R}\n")
            return True

        for c in calls:
            output = _exec_call(c)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": output})

    print(f"\n{RED}[stopped] hit MAX_STEPS={MAX_STEPS} tool round-trips{R}\n")
    return True


def _exec_call(c, indent=""):
    """Execute one tool call: print, approve, checkpoint, run. Returns output string."""
    name = c["name"]
    try:
        args = json.loads(c["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}

    print(f"\n{indent}{TEAL}◆{R} {BOLD}{name}{R}  {GREY}{_preview_call(name, args).splitlines()[0]}{R}")

    if name not in TOOLS:
        print(f"{indent}  {GREY}⎿{R} {RED}unknown tool{R}")
        return f"[error] unknown tool: {name}"

    verdict = approve(name, args)
    if verdict is not True:
        print(f"{indent}  {GREY}⎿{R} {RED}✗ {verdict}{R}")
        return f"[denied] {verdict}"

    # Snapshot file state before a mutating edit so /undo can revert it.
    snap = _snapshot(args.get("path", "")) if name in ("write_file", "edit_file") else None
    try:
        output = TOOLS[name](**args)
    except TypeError as exc:
        output = f"[error] bad arguments: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface to the model
        output = f"[error] {type(exc).__name__}: {exc}"
    if snap is not None and not str(output).startswith("[error]"):
        push_checkpoint(args.get("path", ""), snap, name)
    preview = (output.splitlines()[0] if output else "(no output)")[:100]
    is_err = str(output).startswith("[error]")
    print(f"{indent}  {GREY}⎿{R} {(RED if is_err else GREY)}{preview}{R}")

    if len(output) > MAX_TOOL_OUTPUT:
        output = output[:MAX_TOOL_OUTPUT] + f"\n…[truncated, {len(output)} chars total]"
    return output


# ── Sub-agents / task delegation ──────────────────────────────────────────────
def run_subagent(description):
    """Run a focused nested agent on `description`; return its final answer text.

    The sub-agent has every tool except `task` itself (so it can't recurse),
    shares the approval gate / sandbox / checkpoints, and reports indented
    progress. Only its final summary goes back to the parent.
    """
    sub_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] != "task"]
    msgs = [
        {"role": "system", "content":
            f"You are a focused sub-agent of {NAME}. Complete the delegated task using your "
            f"tools in {os.getcwd()}, then reply with a concise summary of what you found or "
            f"changed. Be thorough but terse."},
        {"role": "user", "content": description},
    ]
    spinner = Spinner()
    print(f"\n  {BLUE}⤷ sub-agent started{R} {DIM}{description[:70]}{R}")
    final = ""
    for _ in range(MAX_STEPS):
        spinner.start()
        res = stream_turn(msgs, spinner, tools=sub_schemas)
        if res is None:
            return "[sub-agent error] transport failure"
        _account(res.get("usage"))
        calls = res["tool_calls"]
        am = {"role": "assistant", "content": res["content"] or ""}
        if calls:
            am["tool_calls"] = [{"id": c["id"], "type": "function",
                                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                                for c in calls]
        msgs.append(am)
        if not calls:
            final = res["content"] or ""
            break
        for c in calls:
            out = _exec_call(c, indent="    ")
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    print(f"  {BLUE}⤷ sub-agent done{R}")
    return final or "(sub-agent produced no result)"


def tool_task(description, **_ignored):
    return run_subagent(description)


TOOLS["task"] = tool_task
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "task",
    "description": ("Delegate a self-contained subtask to a focused sub-agent that has the "
                    "same tools and returns a concise result. Use for multi-step research or "
                    "work you want handled in one shot (e.g. 'find and summarize all places "
                    "that read config')."),
    "parameters": {"type": "object", "properties": {
        "description": {"type": "string", "description": "The subtask, with enough context to act on"},
    }, "required": ["description"]},
}})


# ── Session persistence / resume ──────────────────────────────────────────────
def _user_id():
    """Stable per-user id so sessions never mix on a shared machine.

    Uses HERA_USER (e.g. the user's email) if set, else a hash of the API key
    (each Open WebUI user has their own key → their own session store)."""
    u = _env("HERA_USER")
    if u:
        return re.sub(r"[^A-Za-z0-9._@-]", "_", u)[:64]
    if API_KEY:
        return "key-" + hashlib.sha256(API_KEY.encode()).hexdigest()[:12]
    return "default"


USER_ID = _user_id()
SESSIONS_DIR = os.path.join(
    os.path.expanduser(_env("HERA_SESSIONS_DIR", default="~/.config/hera/sessions")),
    USER_ID)
CURRENT_SESSION = {"id": None, "created": None}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_session_id():
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def save_session(messages):
    """Persist the conversation (skips trivial sessions). Best-effort."""
    if not CURRENT_SESSION["id"] or len(messages) <= 1:
        return
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        state = {
            "id": CURRENT_SESSION["id"],
            "created": CURRENT_SESSION["created"],
            "updated": _now(),
            "cwd": os.getcwd(),
            "model": MODEL,
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


def load_session(sid):
    """Resolve `sid` (exact id, prefix, or '__latest__') to a saved session."""
    sessions = list_sessions()
    if not sessions:
        return None
    if sid in ("__latest__", "", None):
        return sessions[0]
    for s in sessions:
        if s.get("id") == sid:
            return s
    for s in sessions:
        if s.get("id", "").startswith(sid):
            return s
    return None


def _first_user(messages):
    for m in messages:
        if m.get("role") == "user":
            return " ".join((m.get("content") or "").split())[:56]
    return "(empty)"


def print_sessions():
    sessions = list_sessions()
    if not sessions:
        print(f"\n{DIM}no saved sessions in {SESSIONS_DIR}{R}\n")
        return
    print(f"\n{DIM}saved sessions (newest first) — resume with: hera --resume <id>{R}")
    for s in sessions[:20]:
        msgs = s.get("messages", [])
        nturns = sum(1 for m in msgs if m.get("role") == "user")
        print(f"  {BOLD}{s.get('id','?')}{R}  {DIM}{s.get('updated','?')}  "
              f"{nturns} turn(s)  {s.get('tokens',{}).get('total',0)} tok  "
              f"· {_first_user(msgs)}{R}")
    print()


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


def _register_tool(name, description, parameters, func, read_only=False):
    TOOLS[name] = func
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": name, "description": description or name,
        "parameters": parameters or {"type": "object", "properties": {}},
    }})
    EXT_TOOLS.add(name)
    if not read_only:
        SIDE_EFFECTS.add(name)


def register_mcp():
    if not os.path.isfile(MCP_CONFIG):
        return []
    try:
        with open(MCP_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{YELL}[mcp] bad config {MCP_CONFIG}: {exc}{R}", file=sys.stderr)
        return []
    servers = cfg.get("mcpServers", cfg if isinstance(cfg, dict) else {})
    loaded = []
    for sname, spec in servers.items():
        if not isinstance(spec, dict) or "command" not in spec:
            continue
        try:
            client = McpClient(sname, spec["command"], spec.get("args"), spec.get("env"))
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
                        "'where is the logic that …' questions."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural-language query"},
            "path":  {"type": "string", "description": "Directory to search (default '.')"},
            "k":     {"type": "integer", "description": "Number of results (default 8)"},
        }, "required": ["query"]},
    }})
    return True


def register_extensions(quiet=False):
    mcp = register_mcp()
    custom = register_custom_tools()
    out = sys.stderr if quiet else sys.stdout   # serve mode keeps stdout JSON-only
    if mcp:
        print(f"{DIM}[ext] loaded {len(mcp)} MCP tool(s): {', '.join(mcp[:6])}"
              f"{'…' if len(mcp) > 6 else ''}{R}", file=out)
    if custom:
        print(f"{DIM}[ext] loaded {len(custom)} custom tool(s): {', '.join(custom)}{R}", file=out)
    if register_semantic_search():
        print(f"{DIM}[ext] semantic_search enabled (embeddings at {EMBED_URL}){R}", file=out)


def close_extensions():
    for c in _mcp_clients:
        c.close()


# ── Project context (Claude-Code-style) ───────────────────────────────────────
CONTEXT_FILES = ("HERA.md", "AGENTS.md", "AGENT.md")


def load_project_context():
    for fn in CONTEXT_FILES:
        p = os.path.join(os.getcwd(), fn)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return fn, f.read()[:8000]
            except OSError:
                pass
    return None, None


def system_prompt():
    base = (
        f"You are {NAME}, an agentic coding assistant running in a terminal. "
        f"You operate in the working directory: {os.getcwd()} (OS: {sys.platform}). "
        "You have tools to list directories, find files by glob, search file contents by "
        "regex, index code definitions (symbols), read files, write files, edit files by "
        "exact string replacement, and run shell commands. A semantic_search tool may also "
        "be available for fuzzy 'where is the code that…' questions. "
        "Use glob/search/symbols to locate relevant code, then read files before changing "
        "anything; make edits with precise old_string/new_string. File edits are revertible "
        "by the user with /undo. For larger self-contained subtasks you may delegate to a "
        "focused sub-agent with the task tool. "
        "Keep prose short — act with tools rather than describing what you would do. "
        "When the task is complete, give a brief summary of what you changed."
    )
    fn, body = load_project_context()
    if body:
        base += (f"\n\nThe project provides a context file ({fn}). Follow its "
                 f"instructions and conventions:\n\n{body}")
    return base


# ── @file mentions & context compaction ───────────────────────────────────────
def expand_mentions(text):
    """Inline the contents of any @path the user references. Returns (text, names)."""
    attached = []
    for tok in re.findall(r"(?<!\S)@([^\s]+)", text):
        p = _resolve(tok)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    attached.append((tok, f.read()[:MAX_READ_BYTES]))
            except OSError:
                pass
    if not attached:
        return text, []
    blocks = "\n\n".join(f"--- {n} ---\n{c}" for n, c in attached)
    return f"{text}\n\n[Attached files]\n{blocks}", [n for n, _ in attached]


def compact_history(messages):
    """Replace the conversation with a model-written summary to free up context."""
    if len(messages) <= 2:
        return "nothing to compact yet"
    convo = []
    for m in messages[1:]:
        c = m.get("content") or ""
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


# ── Banner / help ──────────────────────────────────────────────────────────────
VERSION = "0.3"

_WORDMARK = [
    "█  █  ████  ███    ██ ",
    "█  █  █     █  █  █  █ ",
    "████  ███   ███   ████ ",
    "█  █  █     █ █   █  █ ",
    "█  █  ████  █  █  █  █ ",
]


def _short(path, n=40):
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path if len(path) <= n else "…" + path[-(n - 1):]


def print_banner():
    host = API_URL.replace("http://", "").replace("https://", "").replace("/v1", "")
    fn, _ = load_project_context()
    shades = [SKY, SKY, TEAL, IND, IND]

    print()
    for sh, line in zip(shades, _WORDMARK):
        print(f"  {sh}{line}{R}")
    print(f"  {DIM}agentic coding CLI{R}  {GREY}· v{VERSION} · {MODEL}{R}\n")

    rule = f"  {GREY}{'─' * 50}{R}"

    def row(label, value, vcolor=""):
        print(f"  {ACCENT}▎{R} {DIM}{label:<8}{R}{vcolor}{value}{R}")

    print(rule)
    row("server", host)
    row("cwd", _short(os.getcwd()))
    row("safety", "auto-approve (YOLO)" if YOLO else "approval on edits & bash",
        RED if YOLO else "")
    row("sandbox", sandbox_label())
    if ALLOW_PATTERNS:
        row("allow", f"{len(ALLOW_PATTERNS)} pattern(s)")
    if EXT_TOOLS:
        row("ext", f"{len(EXT_TOOLS)} mcp/custom tool(s)")
    if fn:
        row("context", fn, GREEN)
    row("tools", f"{len(TOOLS)} available")
    print(rule)
    print(f"  {DIM}type a task  ·  {R}{CYAN}@path{R}{DIM} to attach a file  ·  "
          f"{R}{CYAN}/help{R}{DIM} for commands{R}\n")


def print_help():
    print(
        f"\n{DIM}"
        f"  Ask me to build, fix, explain, or refactor code. I work in the\n"
        f"  current directory and ask before editing files or running shell\n"
        f"  commands (unless HERA_YOLO=1). Reference files with @path to attach them.\n\n"
        f"  /undo       revert the last file write/edit I made\n"
        f"  /diff       show the working-tree git diff\n"
        f"  /compact    summarize the conversation to free up context\n"
        f"  /tokens     show token usage this session\n"
        f"  /tools      list the tools I can use\n"
        f"  /allow      list run_bash allow patterns (or: /allow <pattern>)\n"
        f"  /sandbox    show the run_bash sandbox status\n"
        f"  /sessions   list saved sessions (resume with: hera --resume <id>)\n"
        f"  /reasoning  toggle streaming of my thinking\n"
        f"  /cwd        show the working directory\n"
        f"  /new        save current and start a fresh session\n"
        f"  /clear      same as /new (fresh conversation)\n"
        f"  /help       show this message\n"
        f"  /exit       quit  (Ctrl-C or Ctrl-D also work)\n\n"
        f"  start with --resume [ID] / --continue to pick up a past session,\n"
        f"  or --list-sessions to see them.\n"
        f"{R}"
    )


# ── Headless JSON mode (for the VS Code webview) ──────────────────────────────
# `hera --serve` speaks newline-delimited JSON on stdin/stdout so a GUI can drive
# the full agent. stdout carries ONLY JSON events; logs go to stderr.
#
#   in : {"type":"prompt","text":...} | {"type":"approval","decision":"y|a|p|n"}
#        | {"type":"undo"} | {"type":"clear"} | {"type":"exit"}
#   out: ready | reasoning | token | tool_start | approval_request | tool_end
#        | turn_end | info | error
def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _serve_read():
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def _serve_stream(messages):
    try:
        resp = requests.post(
            f"{API_URL}/chat/completions",
            json={"model": MODEL, "messages": messages, "tools": TOOL_SCHEMAS,
                  "tool_choice": "auto", "stream": True,
                  "stream_options": {"include_usage": True}},
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            stream=True, timeout=600,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        _emit({"type": "error", "message": str(exc)})
        return None

    content, tool_calls, usage, finish = [], {}, None, None
    for raw in resp.iter_lines():
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
    return {"content": "".join(content), "finish_reason": finish,
            "tool_calls": [tool_calls[i] for i in sorted(tool_calls)], "usage": usage}


def _serve_approve(name, args):
    if YOLO or name not in SIDE_EFFECTS or name in _always_ok:
        return True
    if name == "run_bash" and bash_allowed(args.get("command", "")):
        return True
    _emit({"type": "approval_request", "name": name,
           "preview": _strip_ansi(_preview_call(name, args)),
           "command": args.get("command", "")})
    while True:
        msg = _serve_read()
        if msg is None:
            return "aborted (input closed)"
        if msg.get("type") == "approval":
            d = msg.get("decision", "n")
            if d == "a" and name == "run_bash":
                ALLOW_PATTERNS.append(" ".join(args.get("command", "").split()))
            elif d == "a":
                _always_ok.add(name)
            elif d == "p" and name == "run_bash" and args.get("command", "").split():
                ALLOW_PATTERNS.append(args["command"].split()[0] + " *")
            return True if d in ("y", "a", "p") else "user declined"


def _serve_exec(c):
    name = c["name"]
    try:
        args = json.loads(c["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    _emit({"type": "tool_start", "name": name,
           "preview": _strip_ansi(_preview_call(name, args))})
    if name not in TOOLS:
        out = f"[error] unknown tool: {name}"
        _emit({"type": "tool_end", "name": name, "error": True, "output": out})
        return out
    verdict = _serve_approve(name, args)
    if verdict is not True:
        out = f"[denied] {verdict}"
        _emit({"type": "tool_end", "name": name, "error": True, "output": out})
        return out
    snap = _snapshot(args.get("path", "")) if name in ("write_file", "edit_file") else None
    try:
        out = TOOLS[name](**args)
    except TypeError as exc:
        out = f"[error] bad arguments: {exc}"
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
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + f"\n…[truncated, {len(out)} chars total]"
    return out


def _serve_run(messages):
    turn = 0
    for _ in range(MAX_STEPS):
        res = _serve_stream(messages)
        if res is None:
            return
        turn += _account(res.get("usage"))
        calls = res["tool_calls"]
        am = {"role": "assistant", "content": res["content"] or ""}
        if calls:
            am["tool_calls"] = [{"id": x["id"], "type": "function",
                                 "function": {"name": x["name"], "arguments": x["arguments"]}}
                                for x in calls]
        messages.append(am)
        if not calls:
            _emit({"type": "turn_end", "content": res["content"] or "",
                   "turn_tokens": turn, "session_tokens": dict(SESSION)})
            return
        for c in calls:
            out = _serve_exec(c)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    _emit({"type": "turn_end", "content": "[stopped: hit MAX_STEPS]",
           "turn_tokens": turn, "session_tokens": dict(SESSION)})


def serve_main():
    if not API_URL:
        _emit({"type": "error", "message": "no server set — set HERA_API_URL"})
        return
    register_extensions(quiet=True)
    CURRENT_SESSION["id"] = new_session_id()
    CURRENT_SESSION["created"] = _now()
    messages = [{"role": "system", "content": system_prompt()}]
    _emit({"type": "ready", "name": NAME, "model": MODEL, "cwd": os.getcwd(),
           "sandbox": sandbox_label(), "tools": list(TOOLS),
           "needs_key": not bool(API_KEY)})
    while True:
        msg = _serve_read()
        if msg is None:
            break
        t = msg.get("type")
        if t == "exit":
            break
        if t == "prompt":
            content, attached = expand_mentions(msg.get("text", ""))
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


def main():
    global HIDE_REASONING

    ap = argparse.ArgumentParser(prog="hera", add_help=True,
                                 description="Hera — agentic coding CLI")
    ap.add_argument("--resume", "-r", nargs="?", const="__latest__", default=None,
                    metavar="ID", help="resume a saved session (latest if no ID)")
    ap.add_argument("--continue", "-c", dest="cont", action="store_true",
                    help="continue the most recent session")
    ap.add_argument("--list-sessions", "-l", action="store_true",
                    help="list saved sessions and exit")
    ap.add_argument("--serve", action="store_true",
                    help="headless JSON mode over stdin/stdout (used by the VS Code extension)")
    args = ap.parse_args()

    if args.serve:
        serve_main()
        return

    if args.list_sessions:
        print_sessions()
        return

    if not API_URL:
        print(f"{RED}[error] no server set. Export HERA_API_URL (and HERA_API_KEY), e.g.:\n"
              f"  export HERA_API_URL=http://<host>:3000/api   # your Open WebUI endpoint\n"
              f"  export HERA_API_KEY=<your personal key>{R}", file=sys.stderr)
        return
    if not API_KEY:
        print(f"{YELL}[warn] no API key set — the server will reject requests with 401.\n"
              f"       export HERA_API_KEY=<key> and re-run.{R}\n", file=sys.stderr)

    register_extensions()

    # Resume or start fresh.
    resume_target = "__latest__" if args.cont else args.resume
    messages = None
    if resume_target is not None:
        s = load_session(resume_target)
        if s:
            messages = s["messages"]
            CURRENT_SESSION["id"] = s["id"]
            CURRENT_SESSION["created"] = s.get("created")
            SESSION.update(s.get("tokens", {}))
            print(f"{DIM}resumed session {s['id']} "
                  f"({sum(1 for m in messages if m.get('role') == 'user')} turns, "
                  f"{SESSION.get('total', 0)} tokens){R}")
        else:
            print(f"{YELL}no matching session — starting fresh{R}")
    if messages is None:
        messages = _start_new_session()

    print_banner()
    spinner = Spinner()

    try:
        _repl(messages, spinner)
    finally:
        save_session(messages)
        close_extensions()


def _repl(messages, spinner):
    global HIDE_REASONING
    while True:
        try:
            user_input = input(f"{ACCENT}{BOLD}❯{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Session ended.{R}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit"):
            print(f"{DIM}Session ended.{R}")
            break
        if cmd in ("/clear", "/new"):
            save_session(messages)
            messages[:] = _start_new_session()
            _always_ok.clear()
            print(f"\n{DIM}Started a fresh session ({CURRENT_SESSION['id']}).{R}\n")
            continue
        if cmd == "/sessions":
            print_sessions()
            continue
        if cmd == "/undo":
            print(f"\n{DIM}{undo_last()}{R}\n")
            continue
        if cmd == "/diff":
            inrepo = subprocess.run("git rev-parse --is-inside-work-tree",
                                    shell=True, capture_output=True, text=True, cwd=os.getcwd())
            if inrepo.returncode != 0:
                print(f"\n{DIM}not a git repository — /diff needs git{R}\n")
                continue
            proc = subprocess.run("git diff --stat && echo '---' && git diff",
                                  shell=True, capture_output=True, text=True, cwd=os.getcwd())
            out = (proc.stdout or "").rstrip() or "(no changes)"
            print(f"\n{DIM}{out[:6000]}{R}\n")
            continue
        if cmd == "/compact":
            print(f"\n{DIM}compacting…{R}")
            print(f"{DIM}{compact_history(messages)}{R}\n")
            save_session(messages)
            continue
        if cmd == "/help":
            print_help()
            continue
        if cmd == "/tokens":
            s = SESSION
            print(f"\n{DIM}session: {s['total']} tokens over {s['requests']} requests "
                  f"(prompt {s['prompt']} / completion {s['completion']}){R}\n")
            continue
        if cmd == "/tools":
            print(f"\n{DIM}tools: {', '.join(TOOLS)}\n"
                  f"  approval required: {', '.join(sorted(SIDE_EFFECTS))}\n"
                  f"  run_bash sandbox: {sandbox_label()}{R}\n")
            continue
        if cmd == "/sandbox":
            print(f"\n{DIM}sandbox: {sandbox_label()}\n"
                  f"  mode={SANDBOX_MODE} kind={SANDBOX_KIND} network={'on' if SANDBOX_NET else 'off'}\n"
                  f"  change with HERA_SANDBOX=bwrap|unshare|none and HERA_SANDBOX_NET=1{R}\n")
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
        if cmd == "/reasoning":
            HIDE_REASONING = not HIDE_REASONING
            state = "hidden" if HIDE_REASONING else "visible"
            print(f"\n{DIM}reasoning is now {state}.{R}\n")
            continue

        content, attached = expand_mentions(user_input)
        if attached:
            print(f"{DIM}  ↳ attached: {', '.join(attached)}{R}")
        messages.append({"role": "user", "content": content})
        try:
            ok = run_agent(messages, spinner)
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n{DIM}(interrupted){R}\n")
            ok = True  # keep history; user can continue
        if not ok:
            messages.pop()  # roll back the failed user turn
        save_session(messages)  # autosave after every turn


if __name__ == "__main__":
    main()
