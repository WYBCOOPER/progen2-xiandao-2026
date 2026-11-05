#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
[[ -f "$ROOT/public_test.py" && -f "$ROOT/run_public_test.sh" ]] || { echo 'Extract into progen2_contest first.' >&2; exit 1; }
cd "$HERE"
sha256sum -c SHA256SUMS
cd "$ROOT"
mkdir -p results
RUN_DIR=$(mktemp -d "results/v22b-compare-$(date +%Y%m%d)-XXXXXX")
finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    if tar -czf "${RUN_DIR}.tar.gz" "$RUN_DIR"; then
        printf '\nBATCH_ARCHIVE=%s.tar.gz\n' "$RUN_DIR"
    else
        echo 'Archive failed; preserve the result directory.' >&2
        [[ "$status" -ne 0 ]] || status=4
    fi
    exit "$status"
}
trap finish EXIT
mkdir -p "$RUN_DIR/diagnostic"
cp -r "$HERE/". "$RUN_DIR/diagnostic/"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
PYTHONUNBUFFERED=1 python -u "$RUN_DIR/diagnostic/compare.py" "$ROOT" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/comparison.log"
