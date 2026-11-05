#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SOURCE_DIR/../.." && pwd)
if [[ ! -f "$PROJECT_ROOT/run_public_test.sh" || ! -f "$PROJECT_ROOT/public_test.py" ]]; then
    printf '%s\n' 'Extract into the existing progen2_contest project directory.' >&2
    exit 1
fi
cd "$SOURCE_DIR"
sha256sum -c SHA256SUMS
cd "$PROJECT_ROOT"
mkdir -p results
RUN_DIR=$(mktemp -d "results/fp32-sdpa-v7-$(date +%Y%m%d)-XXXXXX")
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
mkdir -p "$RUN_DIR/source/models/progen"
cp "$SOURCE_DIR/solution.py" "$RUN_DIR/source/"
cp "$SOURCE_DIR/models/__init__.py" "$RUN_DIR/source/models/"
cp "$SOURCE_DIR/models/progen/"*.py "$RUN_DIR/source/models/progen/"
cp "$SOURCE_DIR/manifest.json" "$SOURCE_DIR/SHA256SUMS" "$SOURCE_DIR/run_test.sh" "$SOURCE_DIR/README.md" "$RUN_DIR/source/"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
python - "$RUN_DIR/runtime_versions.json" <<'PY'
import importlib.metadata, json, platform, sys, torch
versions = {'python': platform.python_version(), 'torch': str(torch.__version__), 'hip': torch.version.hip}
for name in ('triton', 'transformers', 'tokenizers', 'numpy', 'accelerate'):
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
if torch.cuda.is_available():
    versions['device'] = torch.cuda.get_device_name(0)
    versions['device_total_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
with open(sys.argv[1], 'w', encoding='utf-8') as f: json.dump(versions, f, indent=2)
print('INSTALLED_VERSIONS=' + json.dumps(versions))
PY
RESULT_FILE="$RUN_DIR/candidate_public.json" PYTHONUNBUFFERED=1 \
    bash "$PROJECT_ROOT/run_public_test.sh" "$RUN_DIR/source" 2>&1 | tee "$RUN_DIR/candidate.log"

python - "$RUN_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
log = (root / 'candidate.log').read_text(encoding='utf-8', errors='replace')
events = [int(line.split('=', 1)[1]) for line in log.splitlines()
          if line.startswith('COMPILE_EVENTS_DURING_GENERATE=')]
compile_ok = len(events) == 8 and all(n == 0 for n in events[1::2])
result = json.loads((root / 'candidate_public.json').read_text(encoding='utf-8'))
loads = result.get('loads', [])
expected_names = ['load1_b1_l32_g256', 'load2_b8_l32_g256', 'load3_b4_l32_g512', 'load4_b4_l512_g64']
accuracy = (result.get('all_public_references_exact') is True and len(loads) == 4
            and [r.get('load_id') for r in loads] == expected_names
            and [r.get('total') for r in loads] == [256, 2048, 2048, 256]
            and all(r.get('exact_match') is True and r.get('total', 0) > 0
                    and r.get('same') == r.get('total') for r in loads))
identity_ok = 'FP32_CANDIDATE=fp32_sdpa_v7' in log.splitlines()
report = {'candidate': 'fp32_sdpa_v7', 'generate_compile_events': events,
          'expected_generate_calls': 8, 'warmup_events': events[0::2], 'measured_events': events[1::2],
          'compile_activity_check_passed': compile_ok, 'all_public_references_exact': accuracy,
          'candidate_identity_logged': identity_ok, 'experiment_checks_passed': compile_ok and accuracy and identity_ok}
(root / 'compile_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('COMPILE_AUDIT=' + ('PASS' if compile_ok else 'FAIL'))
print('ACCURACY_CHECK=' + ('PASS' if accuracy else 'FAIL'))
print('CANDIDATE_IDENTITY_CHECK=' + ('PASS' if identity_ok else 'FAIL'))
if not report['experiment_checks_passed']:
    print('Keep this evidence; do not treat this run as an accepted optimization.')
    raise SystemExit(2)
PY
