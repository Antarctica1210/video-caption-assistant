#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking for uv..."

if command -v uv &>/dev/null; then
    echo "    uv already installed: $(uv --version)"
else
    echo "    uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &>/dev/null; then
        echo "    uv installed: $(uv --version)"
    else
        echo "[error] uv installation failed. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

echo "==> Syncing project dependencies..."
uv sync

echo ""
echo "Setup complete. Run the app with:"
echo "  uv run main.py --lang zh"
