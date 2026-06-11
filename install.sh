#!/usr/bin/env bash
#
# Hera CLI installer — transparent, inspect it before running.
#
# Usage (the admin gives approved users the download host):
#   HERA_SERVER=http://<host>:8081 bash <(curl -fsSL http://<host>:8081/install.sh)
#
# This script does exactly three things, and nothing hidden:
#   1. checks you have python3 (and installs the 'requests' library if missing)
#   2. downloads the single-file agent (hera.py) to ~/.local/bin/hera
#   3. tells you how to set your endpoint + API key and run it
#
# It does NOT send anything anywhere, and it does NOT need root.
#
set -euo pipefail

SERVER="${HERA_SERVER:-}"
BIN_DIR="${HERA_BIN_DIR:-$HOME/.local/bin}"
DEST="$BIN_DIR/hera"

if [ -z "$SERVER" ]; then
    echo "error: set HERA_SERVER to the download host, e.g." >&2
    echo "  HERA_SERVER=http://<host>:8081 bash <(curl -fsSL http://<host>:8081/install.sh)" >&2
    exit 1
fi

echo "Hera CLI installer"
echo "  download from : $SERVER/hera.py"
echo "  install to    : $DEST"
echo

# 1. Python check
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found. Install Python 3.7+ and re-run." >&2
    exit 1
fi
echo "✓ python3: $(python3 --version 2>&1)"

# 2. requests check / best-effort install
if python3 -c 'import requests' >/dev/null 2>&1; then
    echo "✓ requests already installed"
else
    echo "• installing the 'requests' library (pip --user)…"
    if ! python3 -m pip install --user requests >/dev/null 2>&1; then
        echo "  warning: could not auto-install requests."
        echo "  run this yourself: python3 -m pip install requests" >&2
    fi
fi

# 3. sandbox hint: bubblewrap gives run_bash full filesystem confinement
if command -v bwrap >/dev/null 2>&1; then
    echo "✓ bubblewrap present — run_bash gets full filesystem confinement"
else
    echo "• bubblewrap (bwrap) not found — run_bash will fall back to a weaker"
    echo "  sandbox (network/PID isolation, no filesystem confinement)."
    if command -v apt-get >/dev/null 2>&1; then
        echo "  for full confinement:  sudo apt-get install -y bubblewrap"
    elif command -v dnf >/dev/null 2>&1; then
        echo "  for full confinement:  sudo dnf install -y bubblewrap"
    elif command -v pacman >/dev/null 2>&1; then
        echo "  for full confinement:  sudo pacman -S bubblewrap"
    elif command -v brew >/dev/null 2>&1; then
        echo "  (macOS has no bubblewrap; run_bash uses the weaker fallback or HERA_SANDBOX=none)"
    else
        echo "  install your distro's 'bubblewrap' package for full confinement."
    fi
fi

# 4. download the agent
mkdir -p "$BIN_DIR"
echo "• downloading hera…"
curl -fsSL "$SERVER/hera.py" -o "$DEST"
chmod +x "$DEST"
echo "✓ installed: $DEST"

# 5. pre-write the endpoint into the config file so the user only pastes a key.
#    The proxy (identity endpoint) is the download host with port 8090.
#    HERA_API_URL overrides the derived value if the admin sets it explicitly.
API_URL="${HERA_API_URL:-}"
if [ -z "$API_URL" ]; then
    host_only="${SERVER#http://}"; host_only="${host_only#https://}"; host_only="${host_only%%:*}"
    host_only="${host_only%%/*}"
    API_URL="http://${host_only}:8090/v1"
fi
CONFIG_DIR="${HERA_CONFIG_DIR:-$HOME/.config/hera}"
CONFIG_FILE="$CONFIG_DIR/config.json"
mkdir -p "$CONFIG_DIR"
if [ -f "$CONFIG_FILE" ] && grep -q '"api_key"' "$CONFIG_FILE" 2>/dev/null; then
    echo "✓ existing config kept: $CONFIG_FILE"
else
    printf '{\n  "api_url": "%s"\n}\n' "$API_URL" > "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    echo "✓ endpoint saved: $API_URL  ($CONFIG_FILE)"
fi

# PATH hint
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        echo "note: $BIN_DIR is not on your PATH. Add it with:"
        echo "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
        ;;
esac

cat <<EOF

Done. The endpoint is already configured. Just run:

       hera

On first launch it will ask you to paste your personal API key once
(Open WebUI → Settings → Account → API Keys). After that it never asks again.

  Optional: to keep sessions labelled by who you are, run once:
       hera   # then your key; or set HERA_USER=<your-email> in ~/.bashrc

EOF
