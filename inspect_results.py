from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .config import repo_root_from_file


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in (root / ".chatgpt/operations/neurogolf_optimizer/runs").glob("*/summary.json"):
        data = _load_json(path)
        run = str(path.parent)
        for key in ("accepted", "near_misses", "errors"):
            for row in data.get(key, []):
                if isinstance(row, dict):
                    rows.append({**row, "source": key, "run_dir": run})
    for path in root.glob("task[0-9][0-9][0-9]/experiments/optimizer_runs/*/summary.json"):
        data = _load_json(path)
        run = str(path.parent)
        for key in ("accepted", "near_misses"):
            for row in data.get(key, []):
                if isinstance(row, dict):
                    rows.append({**row, "source": key, "run_dir": run})
        if not data.get("accepted") and not data.get("near_misses") and "task" in data:
            rows.append({"task": data.get("task"), "source": "summary", "run_dir": run, "candidate_cost": data.get("final_incumbent_cost"), "trials": data.get("trials"), "promoted": data.get("promoted")})
    for path in root.glob("task[0-9][0-9][0-9]/experiments/optimizer_runs/*/near_misses.json"):
        for row in _load_json(path) if path.exists() else []:
            if isinstance(row, dict):
                rows.append({**row, "source": "near_misses", "run_dir": str(path.parent)})
    deduped = []
    seen = set()
    for row in rows:
        key = (row.get("task"), row.get("model_path"), row.get("source"), row.get("candidate_cost"), row.get("near_miss_category"), row.get("accepted"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _format_table(rows: list[dict]) -> str:
    fields = ["task", "candidate_cost", "cost_reduction", "delta_points", "public_wrong", "arc_gen_wrong", "fresh", "family", "model_path", "accepted", "near_miss_category", "promoted"]
    lines = ["\t".join(fields)]
    for row in rows:
        fams = row.get("families") or []
        family = ",".join(fams) if isinstance(fams, list) else str(fams)
        fresh = ""
        if row.get("fresh_right") is not None or row.get("fresh_total") is not None:
            fresh = f"{row.get('fresh_right')}/{row.get('fresh_total')}"
        values = [
            row.get("task"),
            row.get("candidate_cost"),
            row.get("cost_reduction"),
            row.get("delta_points"),
            row.get("public_wrong"),
            row.get("arc_gen_wrong"),
            fresh,
            family,
            row.get("model_path"),
            row.get("accepted"),
            row.get("near_miss_category"),
            row.get("promoted"),
        ]
        lines.append("\t".join("" if v is None else str(v) for v in values))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect NeuroGolf optimizer runs")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--task")
    parser.add_argument("--accepted", action="store_true")
    parser.add_argument("--near-misses", action="store_true")
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    root = repo_root_from_file()
    rows = _rows(root)
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]
    if args.accepted:
        rows = [r for r in rows if r.get("accepted") or r.get("source") == "accepted"]
    if args.near_misses:
        rows = [r for r in rows if r.get("near_miss") or r.get("source") == "near_misses"]
    if args.errors:
        rows = [r for r in rows if r.get("source") == "errors" or "error" in r]
    rows.sort(key=lambda r: r.get("delta_points") or 0, reverse=True)
    rows = rows[: args.top]
    total_gain = sum(r.get("delta_points") or 0.0 for r in rows if r.get("accepted"))
    promoted = sum(1 for r in rows if r.get("promoted"))
    dry = sum(1 for r in rows if r.get("accepted") and not r.get("promoted"))
    if args.csv:
        with Path(args.csv).open("w", newline="") as fh:
            fieldnames = sorted({k for row in rows for k in row.keys()})
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        print(json.dumps({"total_accepted_score_gain": total_gain, "promoted": promoted, "accepted_not_promoted": dry, "rows": rows}, indent=2, sort_keys=True))
    else:
        print(f"total_accepted_score_gain={total_gain:.6f} promoted={promoted} accepted_not_promoted={dry}")
        print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
