"""Command line entry point: ``python -m trustfuse <command>``.

    prepare   convert raw dataset archives (data/raw) into data/processed
    verify    check processed data checksums
    train     train every model of one or more experiment configs (resumable)
    evaluate  calibrate + attack grid + statistics for a config (writes a result set)
    run       train then evaluate (what the API's job worker executes)
    report    regenerate docs/RESULTS.md and docs/figures from the canonical result sets
    smoke     end-to-end self test used by config-everything
    serve     start the API server
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import paths


def _configs(names: list[str]):
    from .experiments.config import ExperimentConfig
    out = []
    for n in names:
        p = Path(n)
        if not p.exists():
            p = paths.CONFIGS_DIR / (n if n.endswith(".yaml") else f"{n}.yaml")
        out.append(ExperimentConfig.from_yaml(p))
    return out


def cmd_prepare(args) -> int:
    from .data.raw import convert_dalia, convert_wesad, write_manifest
    raw = paths.RAW_DIR
    if args.dataset in ("wesad", "all"):
        src = raw / "WESAD.zip"
        out = paths.PROCESSED_DIR / "wesad"
        write_manifest("wesad", out, convert_wesad(src, out), src)
    if args.dataset in ("dalia", "all"):
        src = raw / "data.zip"
        out = paths.PROCESSED_DIR / "dalia"
        write_manifest("dalia", out, convert_dalia(src, out), src)
    return 0


def cmd_verify(_args) -> int:
    from .data.raw import verify_manifest
    problems = []
    for ds in ("wesad", "dalia"):
        problems += verify_manifest(paths.PROCESSED_DIR / ds)
    for p in problems:
        print("PROBLEM:", p)
    print("processed data OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_train(args) -> int:
    from .experiments.training import train_many
    train_many(_configs(args.configs), workers=args.workers, log=lambda s: print(s, flush=True))
    return 0


def cmd_evaluate(args) -> int:
    from .experiments.runner import evaluate_experiment
    for cfg in _configs(args.configs):
        out = evaluate_experiment(cfg, workers=args.workers, log=lambda s: print(s, flush=True),
                                  run_id=args.run_id)
        print(f"results -> {out}", flush=True)
    return 0


def cmd_run(args) -> int:
    from .experiments.runner import run_experiment
    for cfg in _configs(args.configs):
        out = run_experiment(cfg, log=lambda s: print(s, flush=True))
        print(f"results -> {out}", flush=True)
    return 0


def cmd_report(_args) -> int:
    from .evaluation.report_md import write_readme_headline, write_results_md
    print(f"wrote {write_results_md()}")
    readme = write_readme_headline()
    if readme:
        print(f"updated key findings in {readme}")
    return 0


def cmd_smoke(_args) -> int:
    from .service.smoke import run_smoke
    t0 = time.time()
    report = run_smoke()
    print(json.dumps(report, indent=2))
    print(f"smoke test {'PASSED' if report['ok'] else 'FAILED'} in {time.time() - t0:.1f}s")
    return 0 if report["ok"] else 1


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("trustfuse.api.app:app", host=args.host, port=args.port, workers=1, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trustfuse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", choices=["wesad", "dalia", "all"], default="all")
    p.set_defaults(fn=cmd_prepare)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    for name, fn in (("train", cmd_train), ("evaluate", cmd_evaluate), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="+", help="config names in configs/ or YAML paths")
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--run-id", default=None)
        p.set_defaults(fn=fn)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
