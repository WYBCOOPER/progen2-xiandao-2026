#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SOURCE_DIR/../.." && pwd)
if [[ ! -f "$PROJECT_ROOT/run_public_test.sh" || ! -f "$PROJECT_ROOT/public_test.py" ]]; then
    printf '%s\n' 'Please extract this package into the progen2_contest project directory.' >&2
    exit 1
fi
cd "$SOURCE_DIR"
sha256sum -c SHA256SUMS
cd "$PROJECT_ROOT"
mkdir -p results
RUN_DIR=$(mktemp -d "results/fp32-decode-step-v2-$(date +%Y%m%d)-XXXXXX")
mkdir -p "$RUN_DIR/source/models/progen"
cp "$SOURCE_DIR/solution.py" "$RUN_DIR/source/"
cp "$SOURCE_DIR/models/__init__.py" "$RUN_DIR/source/models/"
cp "$SOURCE_DIR/models/progen/"*.py "$RUN_DIR/source/models/progen/"
cp "$SOURCE_DIR/manifest.json" "$SOURCE_DIR/SHA256SUMS" "$RUN_DIR/source/"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
RESULT_FILE="$RUN_DIR/candidate_public.json" PYTHONUNBUFFERED=1 \
    bash "$PROJECT_ROOT/run_public_test.sh" "$RUN_DIR/source" 2>&1 | tee "$RUN_DIR/candidate.log"
printf '\nRESULT_DIRECTORY=%s\n' "$RUN_DIR"
