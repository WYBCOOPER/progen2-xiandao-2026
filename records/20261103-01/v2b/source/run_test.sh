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
RUN_DIR=$(mktemp -d "results/fp32-decode-step-v2b-$(date +%Y%m%d)-XXXXXX")
mkdir -p "$RUN_DIR/source/models/progen"
cp "$SOURCE_DIR/solution.py" "$RUN_DIR/source/"
cp "$SOURCE_DIR/models/__init__.py" "$RUN_DIR/source/models/"
cp "$SOURCE_DIR/models/progen/"*.py "$RUN_DIR/source/models/progen/"
cp "$SOURCE_DIR/manifest.json" "$SOURCE_DIR/SHA256SUMS" "$RUN_DIR/source/"
printf 'RESULT_DIRECTORY=%s\n' "$RUN_DIR"
python - "$RUN_DIR/runtime_versions.json" <<'PY'
import importlib, json, platform, sys
versions={"python": platform.python_version()}
for name in ("torch", "triton", "transformers", "tokenizers", "numpy", "accelerate"):
    try:
        module=importlib.import_module(name)
        versions[name]=str(getattr(module,"__version__","unknown"))
        if name=="torch": versions["hip"]=module.version.hip
    except ImportError:
        versions[name]="not installed"
with open(sys.argv[1],"w",encoding="utf-8") as f: json.dump(versions,f,indent=2)
print("INSTALLED_VERSIONS="+json.dumps(versions))
PY
RESULT_FILE="$RUN_DIR/candidate_public.json" PYTHONUNBUFFERED=1 \
    bash "$PROJECT_ROOT/run_public_test.sh" "$RUN_DIR/source" 2>&1 | tee "$RUN_DIR/candidate.log"
printf '\nRESULT_DIRECTORY=%s\n' "$RUN_DIR"

python - "$RUN_DIR/candidate.log" "$RUN_DIR/compile_audit.json" <<'PY'
import json, pathlib, sys
lines=pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
events=[int(line.split("=",1)[1]) for line in lines if line.startswith("COMPILE_EVENTS_DURING_GENERATE=")]
valid=len(events)==8 and all(n==0 for n in events[1::2])
report={"generate_compile_events":events,"expected_generate_calls":8,"warmup_events":events[0::2],"measured_events":events[1::2],"compile_activity_check_passed":valid}
pathlib.Path(sys.argv[2]).write_text(json.dumps(report,indent=2),encoding="utf-8")
print("COMPILE_AUDIT="+("PASS" if valid else "FAIL: do not use these timings as performance evidence"))
if not valid: raise SystemExit(2)
PY
