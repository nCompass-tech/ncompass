#!/usr/bin/env bash
# Build the optimized kernel (runner/kernel/) as the `hstu_runner` package.
# Registers ops under torch.ops.hstu_opt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REF_CPP_DIR="$REPO_ROOT/generative-recommenders/generative_recommenders/ops/cpp"
PIP="${PIP:-pip}"
PYTHON="${PYTHON:-python}"

if [ ! -d "$SCRIPT_DIR/kernel" ] || [ -z "$(ls -A "$SCRIPT_DIR/kernel/"*.h 2>/dev/null)" ]; then
    echo "Error: fa3/kernel/ not populated. Kernel source should be git-tracked."
    echo "  Check that the fa3/kernel/ directory exists in the repository."
    exit 1
fi

# --- 1. Ensure cutlass submodule is initialized ---
if [ ! -f "$REF_CPP_DIR/cutlass/include/cutlass/cutlass.h" ]; then
    echo "Initializing cutlass submodule..."
    cd "$REPO_ROOT/generative-recommenders"
    git submodule update --init generative_recommenders/ops/cpp/cutlass
    cd "$SCRIPT_DIR"
fi

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
if [ ! -f "$SCRIPT_DIR/hstu_runner/__init__.py" ]; then
    mkdir -p "$SCRIPT_DIR/hstu_runner"
    touch "$SCRIPT_DIR/hstu_runner/__init__.py"
fi

echo "Building hstu_runner (optimized kernel)..."
cd "$SCRIPT_DIR"
$PIP install -e . --no-build-isolation 2>&1
echo "Done. Verify: $PYTHON -c 'import hstu_runner._C; print(\"hstu_runner OK\")'"
