#!/usr/bin/env bash
# Build the reference (unmodified) kernel as the `hstu` package.
# Registers ops under torch.ops.hstu.
#
# Uses fa3/reference/setup.py which builds from the generative-recommenders
# source tree via absolute paths — the upstream repo is never modified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REF_CPP_DIR="$REPO_ROOT/generative-recommenders/generative_recommenders/ops/cpp"
PIP="${PIP:-pip}"
PYTHON="${PYTHON:-python}"

# --- 1. Ensure cutlass submodule is initialized ---
if [ ! -f "$REF_CPP_DIR/cutlass/include/cutlass/cutlass.h" ]; then
    echo "Initializing cutlass submodule..."
    cd "$REPO_ROOT/generative-recommenders"
    git submodule update --init generative_recommenders/ops/cpp/cutlass
    cd "$SCRIPT_DIR"
fi

# --- 2. Build ---
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=FALSE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE

# --- Ensure hstu package dir exists (needed by find_packages / setup.py develop) ---
if [ ! -f "$SCRIPT_DIR/reference/hstu/__init__.py" ]; then
    mkdir -p "$SCRIPT_DIR/reference/hstu"
    touch "$SCRIPT_DIR/reference/hstu/__init__.py"
fi

echo "Building hstu (reference kernel)..."
cd "$SCRIPT_DIR/reference"
$PIP install -e . --no-build-isolation 2>&1
echo "Done. Verify: $PYTHON -c 'import hstu._C; print(\"hstu OK\")'"
