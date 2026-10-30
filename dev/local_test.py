"""Local NVIDIA development checks. Run from an extracted contest package.

Place this file at <contest-root>/dev/local_test.py. It imports validation
functions from the unchanged public_test.py. Outputs are development records,
not official DCU scores. No dependencies are installed by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import sys
import time
import traceback


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_records(reference: dict, candidate: dict) -> list[dict]:
    for record in (reference, candidate):
        if record.get("purpose") != "local_development":
            raise ValueError("Compare only local_development records")
    for key in ("cases_sha256", "torch", "cuda", "host_id", "device", "precision"):
        if reference.get(key) != candidate.get(key):
            raise ValueError("Comparison conditions differ: " + key)
    a_rows = reference.get("loads", [])
    b_rows = candidate.get("loads", [])
    if not a_rows or len(a_rows) != len(b_rows):
        raise ValueError("Compared workloads differ")
    result = []
    for a, b in zip(a_rows, b_rows):
        for key in ("load_id", "batch_size", "prefix_tokens", "new_tokens", "input_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError("Workload conditions differ: " + key)
        batch, length = b["batch_size"], b["new_tokens"]
        for row in (a, b):
            values = row.get("outputs")
            if not isinstance(values, list) or len(values) != batch:
                raise ValueError("Output batch differs")
            if any(not isinstance(v, str) or len(v) != length for v in values):
                raise ValueError("Output lengths differ")
        same = sum(x == y for ra, rb in zip(a["outputs"], b["outputs"]) for x, y in zip(ra, rb))
        total = batch * length
        result.append({"load_id": b["load_id"], "same": same, "total": total,
                       "consistency_rate": same / total, "exact_match": same == total})
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--submission", required=True, help="Directory containing solution.py")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--compare", type=Path, help="An earlier local baseline JSON")
    p.add_argument("--case-id", help="One load id; omit to run all four")
    p.add_argument("--new-tokens", type=int, help="Short development length, at most original G")
    p.add_argument("--state-check", action="store_true", help="Compare A then B then A on one object")
    args = p.parse_args()
    root = args.root.resolve()
    official_path = root / "public_test.py"
    cases_path = root / "public_test_data/public_cases.json"
    submission = Path(args.submission)
    if not submission.is_absolute():
        submission = root / submission
    submission = submission.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output filename; preserving " + str(output))
    if not official_path.is_file() or not cases_path.is_file():
        raise FileNotFoundError("Use --root to select the extracted official contest package")
    if not (submission / "solution.py").is_file():
        raise FileNotFoundError("Missing solution.py in " + str(submission))

    spec = importlib.util.spec_from_file_location("contest_public_validation", official_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import official validation functions")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)  # main() is not invoked or modified.
    official.configure_runtime()
    import torch
    from tokenizers import Tokenizer

    if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("This development entry needs NVIDIA CUDA PyTorch. Use official tools on DCU.")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    assets = root / "progen2"
    tokenizer_path = assets / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["loads"]
    selected = [c for c in cases if args.case_id is None or c["id"] == args.case_id]
    if not selected:
        raise ValueError("Unknown --case-id")
    for c in selected:
        g = c["new_tokens"] if args.new_tokens is None else args.new_tokens
        if not 1 <= g <= c["new_tokens"]:
            raise ValueError("--new-tokens must be positive and at most original G")

    print("LOCAL DEVELOPMENT ONLY: outputs and times are not official DCU scores.")
    solution = official.load_solution(submission, assets / "checkpoints/progen2-medium", tokenizer_path, "cuda:0")
    precision = str(next(solution.model.parameters()).dtype) if hasattr(solution, "model") else "unknown"

    def generate(prefixes, batch, prefix_length, g):
        official.validate_prefixes(tokenizer, prefixes, batch, prefix_length)
        if hasattr(solution, "reset"):
            solution.reset()
        with torch.inference_mode():
            values = solution.generate(prefixes, g)
        official.synchronize(device, torch)
        return official.validate_outputs(values, batch, g)

    rows = []
    for case in selected:
        b, l = case["batch_size"], case["prefix_tokens"]
        g = case["new_tokens"] if args.new_tokens is None else args.new_tokens
        generate(case["warmup_prefixes"], b, l, g)
        official.synchronize(device, torch)
        started = time.perf_counter()
        values = generate(case["prefixes"], b, l, g)
        development_seconds = time.perf_counter() - started
        state_exact = None
        if args.state_check:
            generate(case["warmup_prefixes"], b, l, g)
            repeated = generate(case["prefixes"], b, l, g)
            state_exact = repeated == values
        rows.append({"load_id": case["id"], "batch_size": b, "prefix_tokens": l,
                     "new_tokens": g,
                     "input_sha256": hashlib.sha256(json.dumps(case["prefixes"]).encode()).hexdigest(),
                     "development_time_s": development_seconds, "outputs": values,
                     "state_check_exact": state_exact})
        print(case["id"], "B/L/G=", b, l, g, "format=PASS", "state=", state_exact)

    record = {"purpose": "local_development", "official_score": False,
              "host_id": hashlib.sha256(platform.node().encode()).hexdigest()[:16],
              "cases_sha256": sha256(cases_path), "solution_sha256": sha256(submission / "solution.py"),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(device), "precision": precision,
              "loads": rows}
    # Configuration matching precedes any cross-version numerical comparison.
    if args.compare:
        compared = args.compare if args.compare.is_absolute() else root / args.compare
        record["comparison_to_local_baseline"] = compare_records(
            json.loads(compared.read_text(encoding="utf-8")), record)
        for row in record["comparison_to_local_baseline"]:
            print("LOCAL BASELINE COMPARISON", row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved local record:", output)
    state_ok = all(row["state_check_exact"] is not False for row in rows)
    comparison_ok = all(row["exact_match"] for row in record.get("comparison_to_local_baseline", []))
    if not state_ok or not comparison_ok:
        print("LOCAL_CHECK_FAILED: inspect saved outputs; no official score is implied.")
        return 3
    print("LOCAL_CHECK_PASSED: only the selected local inputs/lengths were checked.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
