#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
mkdir -p results
BATCH=$(mktemp -d "results/v9-next-two-$(date +%Y%m%d)-XXXXXX")
overall=0
printf 'candidate\texit_status\n' > "$BATCH/status.tsv"
for name in fp32_rope_op_v10 fp32_greedy_v11; do
    echo "START_CANDIDATE=$name"
    set +e
    bash "candidates/$name/run_test.sh" 2>&1 | tee "$BATCH/$name.log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\t%s\n' "$name" "$status" >> "$BATCH/status.tsv"
    if [[ "$status" -ne 0 ]]; then overall=1; fi
    archive=$(sed -n 's/^RESULT_ARCHIVE=//p' "$BATCH/$name.log" | tail -n 1)
    if [[ -n "$archive" && -f "$archive" ]]; then
        cp -- "$archive" "$BATCH/"
    else
        echo "Missing child archive for $name" >&2
        overall=1
    fi
done
printf '%s\n' "$overall" > "$BATCH/exit_status.txt"
tar -czf "${BATCH}.tar.gz" "$BATCH"
printf 'BATCH_ARCHIVE=%s.tar.gz\n' "$BATCH"
exit "$overall"
