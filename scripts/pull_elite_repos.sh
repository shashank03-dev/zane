#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$ROOT_DIR/external"
mkdir -p "$EXTERNAL_DIR"

repos=(
  "https://github.com/pschwllr/MolecularTransformer|MolecularTransformer"
  "https://github.com/gcorso/DiffDock|DiffDock"
  "https://github.com/DeepGraphLearning/torchdrug|torchdrug"
  "https://github.com/aqlaboratory/openfold|openfold"
  "https://github.com/openmm/openmm|openmm"
  "https://github.com/CASPistachio/pistachio|pistachio"
)

echo "Syncing elite repositories into external/..."

for entry in "${repos[@]}"; do
  url="${entry%%|*}"
  name="${entry##*|}"
  dst="$EXTERNAL_DIR/$name"

  if [[ -d "$dst/.git" ]]; then
    echo "[update] $name"
    git -C "$dst" pull --ff-only || echo "[warn] pull failed for $name"
  elif [[ -e "$dst" ]]; then
    echo "[skip] $name exists but is not a git repo"
  else
    echo "[clone] $name"
    git clone --depth 1 "$url" "$dst" || echo "[warn] clone failed for $name ($url)"
  fi
done

echo "Done."
