#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
if [[ ! -f "$ROOT/public_test.py" || ! -f "$ROOT/run_public_test.sh" ]]; then echo 'Extract into progen2_contest first.' >&2; exit 1; fi
cd "$HERE"
sha256sum -c SHA256SUMS
cd "$ROOT"
mkdir -p results
RUN_DIR=$(mktemp -d "results/v6-repeat-$(date +%Y%m%d)-XXXXXX")
finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    if tar -czf "${RUN_DIR}.tar.gz" "$RUN_DIR"; then
        printf '\nRESULT_ARCHIVE=%s.tar.gz\n' "$RUN_DIR"
    else
        echo 'Archive creation failed; keep the result directory.' >&2
        if [[ "$status" -eq 0 ]]; then status=4; fi
    fi
    exit "$status"
}
trap finish EXIT
cp -R "$HERE" "$RUN_DIR/diagnostic"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
python -u "$RUN_DIR/diagnostic/benchmark.py" --root "$ROOT" --output "$RUN_DIR" 2>&1 | tee "$RUN_DIR/diagnostic.log"
