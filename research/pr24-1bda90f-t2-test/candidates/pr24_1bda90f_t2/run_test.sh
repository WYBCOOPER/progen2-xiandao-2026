#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SOURCE_DIR/../.." && pwd)
if [[ ! -f "$PROJECT_ROOT/run_public_test.sh" || ! -f "$PROJECT_ROOT/public_test.py" ]]; then
    printf '%s\n' 'Extract this package into the existing progen2_contest project directory.' >&2
    exit 1
fi
cd "$SOURCE_DIR"
sha256sum -c SHA256SUMS
cd "$PROJECT_ROOT"
mkdir -p results
RUN_DIR=$(mktemp -d "results/pr24-1bda90f-t2-$(date +%Y%m%d)-XXXXXX")
finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    printf '\nRESULT_DIRECTORY=%s\n' "$RUN_DIR"
    if tar -czf "${RUN_DIR}.tar.gz" "$RUN_DIR"; then
        printf 'RESULT_ARCHIVE=%s.tar.gz\n' "$RUN_DIR"
    else
        printf '%s\n' 'Archive creation failed; keep the result directory.' >&2
        if [[ "$status" -eq 0 ]]; then status=4; fi
    fi
    exit "$status"
}
trap finish EXIT
mkdir -p "$RUN_DIR/source"
cp -R "$SOURCE_DIR/." "$RUN_DIR/source/"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
python - "$RUN_DIR/runtime_versions.json" <<'PY'
import importlib.metadata, json, platform, sys, torch
versions={'python':platform.python_version(),'torch':str(torch.__version__),'hip':torch.version.hip}
for name in ('triton','transformers','tokenizers','numpy','accelerate'):
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
with open(sys.argv[1],'w',encoding='utf-8') as f: json.dump(versions,f,indent=2)
print('INSTALLED_VERSIONS='+json.dumps(versions))
PY
RESULT_FILE="$RUN_DIR/candidate_public.json" PYTHONUNBUFFERED=1 \
    bash "$PROJECT_ROOT/run_public_test.sh" "$RUN_DIR/source" 2>&1 | tee "$RUN_DIR/candidate.log"

python - "$RUN_DIR" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
result=json.loads((root/'candidate_public.json').read_text(encoding='utf-8'))
log=(root/'candidate.log').read_text(encoding='utf-8',errors='replace')
loads=result.get('loads',[])
accuracy=(result.get('all_public_references_exact') is True and len(loads)==4
          and all(x.get('exact_match') is True and x.get('total',0)>0
                  and x.get('same')==x.get('total') for x in loads))
setting_logged='reduced-precision reduction: fp16/bf16 = False' in log
compile_disabled_logged='torch.compile 已关闭（COMPILE_MODEL=False）' in log
report={'purpose':'pr24_t2_precision_diagnosis','upstream_commit':'1bda90f824b34bfe4a522bef14f6d446661839ad',
        'requested_config':{'WEIGHT_DTYPE':'fp16','HALF_WEIGHT_SCOPE':'all','ALLOW_FP16_REDUCED_PRECISION':False,'COMPILE_MODEL':False},
        'setting_success_logged':setting_logged,'compile_disabled_logged':compile_disabled_logged,'all_public_references_exact':accuracy,
        'dynamo_compile_events_monitored':False,'dcu_backend_effectiveness_confirmed':False,
        't2_checks_passed':accuracy and setting_logged and compile_disabled_logged}
(root/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('T2_CHECK='+('PASS' if accuracy and setting_logged and compile_disabled_logged else 'FAIL'))
if not accuracy: print('Accuracy mismatch: keep this evidence; do not treat as an accepted optimization.')
if not setting_logged: print('Expected backend setting confirmation missing: this is not a confirmed T2 configuration.')
if not compile_disabled_logged: print('Expected COMPILE_MODEL=False confirmation missing.')
if not accuracy or not setting_logged or not compile_disabled_logged: raise SystemExit(2)
PY
