#!/usr/bin/env bash
# Clone nanoGPT at a PINNED commit into third_party/ (gitignored).
#
# We vendor model.py from this checkout into swarm/model/gpt.py (Step 6) and may
# reuse its data/*/prepare.py as a reference. Pinning makes the model source
# reproducible; bump PINNED_SHA deliberately, never float on upstream master.
set -euo pipefail

PINNED_SHA="3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
DEST="third_party/nanoGPT"

if [ -d "$DEST/.git" ]; then
  echo "[clone_nanogpt] $DEST already present at $(git -C "$DEST" rev-parse --short HEAD)"
else
  git clone https://github.com/karpathy/nanoGPT "$DEST"
fi

git -C "$DEST" fetch --depth 1 origin "$PINNED_SHA" 2>/dev/null || git -C "$DEST" fetch origin
git -C "$DEST" checkout -q "$PINNED_SHA"
echo "[clone_nanogpt] checked out $PINNED_SHA"
