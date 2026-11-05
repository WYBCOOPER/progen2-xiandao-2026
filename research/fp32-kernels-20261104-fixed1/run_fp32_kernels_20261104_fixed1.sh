#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT" || exit 1
mkdir -p results
BATCH_DIR=$(mktemp -d "results/fp32-kernels-fixed1-batch-$(date +%Y%m%d)-XXXXXX")
overall=0
finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$BATCH_DIR/exit_status.txt"
    tar -czf "${BATCH_DIR}.tar.gz" "$BATCH_DIR" || exit 4
    printf '\nBATCH_ARCHIVE=%s.tar.gz\n' "$BATCH_DIR"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM
printf 'candidate\texit_status\n' > "$BATCH_DIR/statuses.tsv"
for candidate in fp32_decode_attn_v8b fp32_rope_kv_v9b; do
    printf '\nSTARTING=%s\n' "$candidate"
    if bash "$ROOT/candidates/$candidate/run_test.sh" 2>&1 | tee "$BATCH_DIR/$candidate.log"; then status=0; else status=$?; overall=1; fi
    printf '%s\t%s\n' "$candidate" "$status" >> "$BATCH_DIR/statuses.tsv"
    archive=$(sed -n 's/^RESULT_ARCHIVE=//p' "$BATCH_DIR/$candidate.log" | tail -n 1)
    if [[ -n "$archive" && -f "$archive" ]]; then cp "$archive" "$BATCH_DIR/"; else overall=1; fi
done
exit "$overall"
