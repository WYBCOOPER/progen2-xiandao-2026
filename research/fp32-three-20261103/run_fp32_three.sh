#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT" || exit 1
if [[ ! -f run_public_test.sh ]]; then echo 'Extract into progen2_contest first.' >&2; exit 1; fi
mkdir -p results
BATCH_DIR=$(mktemp -d "results/fp32-three-batch-$(date +%Y%m%d)-XXXXXX")
trap 'exit 130' INT TERM
printf 'candidate\texit_status\n' > "$BATCH_DIR/statuses.tsv"
overall=0
for candidate in fp32_chunk2_v5 fp32_k_transpose_v6 fp32_sdpa_v7; do
    printf '\nSTARTING=%s\n' "$candidate"
    if bash "$ROOT/candidates/$candidate/run_test.sh" 2>&1 | tee "$BATCH_DIR/$candidate.log"; then
        status=0
    else
        status=$?
        overall=1
    fi
    printf '%s\t%s\n' "$candidate" "$status" >> "$BATCH_DIR/statuses.tsv"
    printf 'CANDIDATE_FINISHED=%s EXIT_STATUS=%s\n' "$candidate" "$status"
done
python - "$BATCH_DIR" <<'PY'
from pathlib import Path
import re,sys,tarfile
batch=Path(sys.argv[1]);root=Path.cwd().resolve();results=(root/'results').resolve()
names=['fp32_chunk2_v5','fp32_k_transpose_v6','fp32_sdpa_v7']
with tarfile.open(str(batch)+'.tar.gz','w:gz') as tar:
    tar.add(batch,arcname=batch.as_posix())
    for name in names:
        log=(batch/(name+'.log')).read_text(encoding='utf8',errors='replace')
        paths=re.findall(r'^RESULT_DIRECTORY=(results/[^\r\n]+)$',log,re.M)
        if not paths:continue
        path=Path(paths[-1]);resolved=path.resolve()
        if not resolved.is_relative_to(results) or not path.name.startswith(name.replace('_','-')+'-'):
            raise RuntimeError('Unexpected result path: '+str(path))
        if not path.is_dir():raise RuntimeError('Missing result directory: '+str(path))
        tar.add(path,arcname=path.as_posix())
print('BATCH_RESULT_ARCHIVE='+str(batch)+'.tar.gz')
PY
archive_status=$?
if [[ "$archive_status" -ne 0 ]]; then exit "$archive_status"; fi
exit "$overall"
