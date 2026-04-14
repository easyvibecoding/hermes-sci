#!/usr/bin/env bash
# hermes-sci skill installer — editable pip install of the bundled package.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$SKILL_DIR/package"

echo "installing hermes-sci (editable) from $PKG_DIR"
pip3 install -e "$PKG_DIR"

echo ""
echo "verifying CLI..."
if command -v hermes-sci >/dev/null; then
    hermes-sci --help | head -8
else
    echo "⚠️  'hermes-sci' binary not found in PATH — add \$HOME/.local/bin or your venv bin dir."
fi

echo ""
echo "checking pdflatex + chktex..."
for bin in pdflatex chktex; do
    if command -v $bin >/dev/null; then
        echo "  ✓ $bin: $(command -v $bin)"
    else
        echo "  ✗ $bin not found."
        case $bin in
            pdflatex) echo "    macOS: brew install --cask mactex-no-gui";;
            chktex)   echo "    conda: conda install -c conda-forge chktex";;
        esac
    fi
done

echo ""
echo "all set. try: hermes-sci --help"
