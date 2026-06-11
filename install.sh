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

Done. Set your endpoint + personal key (from your Open WebUI account), then run:

  1. Set them for this shell (edit the values):
       export HERA_API_URL=http://<host>:8090/v1     # the identity proxy
       export HERA_API_KEY=<your personal API key>   # Open WebUI → Settings → Account → API keys
       export HERA_USER=<your-email>                 # optional: keeps your sessions separate

  2. Persist them so new terminals work too (captures what you set above), and reload:
       printf 'export HERA_API_URL=%s\nexport HERA_API_KEY=%s\nexport HERA_USER=%s\n' \\
         "\$HERA_API_URL" "\$HERA_API_KEY" "\$HERA_USER" >> ~/.bashrc && source ~/.bashrc

  3. cd into your project, then run:
       hera

EOF
