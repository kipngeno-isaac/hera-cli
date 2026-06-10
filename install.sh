#!/usr/bin/env bash
#
# Hera CLI installer — transparent, inspect it before running:
#   curl -fsSL http://<HOST>:8081/install.sh
#
# Usage:
#   curl -fsSL http://<HOST>:8081/install.sh | bash
#
# This script does exactly three things, and nothing hidden:
#   1. checks you have python3 (and installs the 'requests' library if missing)
#   2. downloads the single-file agent (hera.py) to ~/.local/bin/hera
#   3. tells you how to set your API key and run it
#
# It does NOT send anything anywhere, and it does NOT need root.
#
set -euo pipefail

SERVER="${HERA_SERVER:-http://<HOST>:8081}"
BIN_DIR="${HERA_BIN_DIR:-$HOME/.local/bin}"
DEST="$BIN_DIR/hera"

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

# 3. download the agent
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

Done. Two steps left:

  1. Set your API key (ask the admin for it — it's LLAMA_API_KEY on the server):
       export HERA_API_KEY=<key>

  2. cd into the project you want to work on, then run:
       hera

EOF
