#!/usr/bin/env bash
# NCU profiling wrapper for the HSTU attention kernel.
#
# Usage:
#   bash fa3/profile_ncu.sh                                 # default config
#   bash fa3/profile_ncu.sh -o my_report                    # custom output name
#   bash fa3/profile_ncu.sh --kernel-regex "FlashAttn"      # custom kernel filter
#   bash fa3/profile_ncu.sh -- --batch-size 256             # pass args to profile_ncu.py
#   bash fa3/profile_ncu.sh -- --kernel ref                 # profile reference kernel
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

OUTPUT_NAME="hstu_profile"
EXTRA_ARGS=()

# Parse wrapper args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --kernel-regex)
            KERNEL_REGEX="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

OUTPUT_DIR="$SCRIPT_DIR/ncu_reports"
mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_NAME"

echo "NCU profiling: $OUTPUT_PATH.ncu-rep"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"

ncu \
    --set full \
    --profile-from-start off \
    -o "$OUTPUT_PATH" \
    --force-overwrite \
    --kernel-id=::device_kernel:\
    $PYTHON "$SCRIPT_DIR/profile_ncu.py" "${EXTRA_ARGS[@]}"

echo ""
echo "Report saved: $OUTPUT_PATH.ncu-rep"
echo "View with: ncu-ui $OUTPUT_PATH.ncu-rep"
