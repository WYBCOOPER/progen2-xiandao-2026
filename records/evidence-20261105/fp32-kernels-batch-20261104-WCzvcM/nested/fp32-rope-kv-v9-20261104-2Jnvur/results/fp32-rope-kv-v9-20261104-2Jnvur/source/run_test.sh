#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
if [[ ! -f "$ROOT/public_test.py" || ! -f "$ROOT/run_public_test.sh" ]]; then echo 'Extract into progen2_contest first.' >&2; exit 1; fi
cd "$HERE"
sha256sum -c SHA256SUMS
cd "$ROOT"
mkdir -p results
RUN_DIR=$(mktemp -d "results/fp32-rope-kv-v9-$(date +%Y%m%d)-XXXXXX")
finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    printf '\nRESULT_DIRECTORY=%s\n' "$RUN_DIR"
    if tar -czf "${RUN_DIR}.tar.gz" "$RUN_DIR"; then
        printf 'RESULT_ARCHIVE=%s.tar.gz\n' "$RUN_DIR"
    else
        echo 'Archive failed; keep the result directory.' >&2
        if [[ "$status" -eq 0 ]]; then status=4; fi
    fi
    exit "$status"
}
trap finish EXIT
mkdir -p "$RUN_DIR/source/models/progen"
cp "$HERE/"*.py "$HERE/README.md" "$HERE/manifest.json" "$HERE/SHA256SUMS" "$HERE/run_test.sh" "$RUN_DIR/source/"
cp "$HERE/models/__init__.py" "$RUN_DIR/source/models/"
cp "$HERE/models/progen/"*.py "$RUN_DIR/source/models/progen/"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
python -u - "$RUN_DIR/runtime_versions.json" "$ROOT" <<'PY'
import hashlib, importlib.metadata, json, platform, pathlib, sys, torch
versions={'python':platform.python_version(),'torch':str(torch.__version__),'hip':torch.version.hip}
for name in ['triton','transformers','tokenizers','numpy','accelerate']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
if torch.cuda.is_available():
    versions['device']=torch.cuda.get_device_name(0)
    versions['device_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
root=pathlib.Path(sys.argv[2])
versions['official_tools_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['public_test.py','run_public_test.sh','baseline/solution.py','check_environment.py','public_test_data/public_cases.json']}
pathlib.Path(sys.argv[1]).write_text(json.dumps(versions,indent=2)+'\n',encoding='utf8')
print('RUNTIME='+json.dumps(versions),flush=True)
assert torch.cuda.is_available() and torch.version.hip is not None,'Requires the DCU/HIP environment'
PY
python -u "$RUN_DIR/source/kernel_check.py" 2>&1 | tee "$RUN_DIR/kernel_check.log"
RESULT_FILE="$RUN_DIR/candidate_public.json" PYTHONUNBUFFERED=1 \
    bash "$ROOT/run_public_test.sh" "$RUN_DIR/source" 2>&1 | tee "$RUN_DIR/candidate.log"
python "$RUN_DIR/source/audit_result.py" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/audit.log"
