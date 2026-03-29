#!/usr/bin/env python3
"""
Train/evaluate the core factorized ablation suite and write aggregated summaries.

The suite runs four experiments:
1. plain MM-BERT baseline
2. factorized v4_03_base_only
3. factorized v4_02_no_mmbert_branch
4. factorized v4_01_style_expr_two_stage

Each run is evaluated on:
- R10 candidates:   ./data/validation_pair_with_cand.json
- full set (Rall):  ./data/validation_pair.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "core_ablation"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    title: str
    entrypoint: str
    config: str
    kind: str


@dataclass(frozen=True)
class EvalProtocol:
    name: str
    data_path: str
    test_with_cand: bool


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "plain_mmbert": ExperimentSpec(
        name="plain_mmbert",
        title="Plain MM-BERT",
        entrypoint="main_mmbbert_yaml.py",
        config="configs/mmbbert/v0_01_validation_baseline.yaml",
        kind="legacy",
    ),
    "v4_03_base_only": ExperimentSpec(
        name="v4_03_base_only",
        title="Factorized v4_03_base_only",
        entrypoint="main_structured_factorized.py",
        config="configs/structured_factorized/v4_03_base_only.yaml",
        kind="factorized",
    ),
    "v4_02_no_mmbert_branch": ExperimentSpec(
        name="v4_02_no_mmbert_branch",
        title="Factorized v4_02_no_mmbert_branch",
        entrypoint="main_structured_factorized.py",
        config="configs/structured_factorized/v4_02_no_mmbert_branch.yaml",
        kind="factorized",
    ),
    "v4_01_style_expr_two_stage": ExperimentSpec(
        name="v4_01_style_expr_two_stage",
        title="Factorized v4_01_style_expr_two_stage",
        entrypoint="main_structured_factorized.py",
        config="configs/structured_factorized/v4_01_style_expr_two_stage.yaml",
        kind="factorized",
    ),
}

PROTOCOLS: Tuple[EvalProtocol, ...] = (
    EvalProtocol(
        name="r10",
        data_path="./data/validation_pair_with_cand.json",
        test_with_cand=True,
    ),
    EvalProtocol(
        name="rall",
        data_path="./data/validation_pair.json",
        test_with_cand=False,
    ),
)

EVAL_RE = re.compile(
    r"\[(?:StructuredEval|EvalSummary)\].*?"
    r"mode=(?P<mode>\S+).*?"
    r"r@1=(?P<r1>\d+\.\d+)\s+"
    r"r@2=(?P<r2>\d+\.\d+)\s+"
    r"r@5=(?P<r5>\d+\.\d+)"
    r"(?:\s+r@10=(?P<r10>\d+\.\d+))?"
    r"(?:\s+r@20=(?P<r20>\d+\.\d+))?\s+"
    r"mrr=(?P<mrr>\d+\.\d+)"
)

FACTORIZED_DIAG_RE = re.compile(
    r"\[StructuredFactorizedScale\].*?"
    r"mmbert_std=(?P<mmbert_std>\d+\.\d+)\s+"
    r"rank_std=(?P<rank_std>\d+\.\d+)\s+"
    r"style_std=(?P<style_std>\d+\.\d+)\s+"
    r"expr_std=(?P<expr_std>\d+\.\d+)\s+"
    r"graph_std=(?P<graph_std>\d+\.\d+)\s+"
    r"union_size=(?P<union_size>\d+\.\d+)\s+"
    r"union_rate=(?P<union_rate>\d+\.\d+)\s+"
    r"gold_union=(?P<gold_union>\d+\.\d+)\s+"
    r"gold_style=(?P<gold_style>\d+\.\d+)\s+"
    r"gold_proto=(?P<gold_proto>\d+\.\d+)\s+"
    r"gold_mmbert=(?P<gold_mmbert>\d+\.\d+)\s+"
    r"delta_abs=(?P<delta_abs>\d+\.\d+)\s+"
    r"top1_flip=(?P<top1_flip>\d+\.\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiments to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        required=True,
        help='GPU groups, e.g. "2,3" "6,7". One worker is created per group.',
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-run and aggregate outputs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional global epoch override for every run.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run evaluation/aggregation from existing checkpoints.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation and only aggregate existing summaries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs even if a summary JSON already exists.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_jobs(jobs: Sequence[Tuple[ExperimentSpec, int]], n_groups: int) -> List[List[Tuple[ExperimentSpec, int]]]:
    groups: List[List[Tuple[ExperimentSpec, int]]] = [[] for _ in range(n_groups)]
    for idx, job in enumerate(jobs):
        groups[idx % n_groups].append(job)
    return groups


def latest_version_dir(run_root: Path) -> Path:
    versions = sorted(
        (run_root / "lightning_logs").glob("version_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not versions:
        raise FileNotFoundError(f"No Lightning version directory under {run_root}")
    return versions[-1]


def latest_final_ckpt(run_root: Path) -> Path:
    ckpts = sorted(
        run_root.glob("lightning_logs/version_*/final.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise FileNotFoundError(f"No final.ckpt found under {run_root}")
    return ckpts[-1]


def latest_test_log(version_dir: Path, protocol: EvalProtocol) -> Path:
    stem = Path(protocol.data_path).stem
    eval_tag = "r10" if protocol.test_with_cand else "rall"
    matches = sorted(
        version_dir.glob(f"test_{stem}_{eval_tag}_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No test log found for protocol={protocol.name} in {version_dir}"
        )
    return matches[-1]


def parse_float_dict(match: re.Match[str]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for key, value in match.groupdict().items():
        if value is None:
            continue
        if key == "mode":
            continue
        res[key] = float(value)
    return res


def parse_eval_log(log_path: Path, kind: str) -> Dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    eval_matches = list(EVAL_RE.finditer(text))
    if not eval_matches:
        raise ValueError(f"Could not parse eval metrics from {log_path}")
    metrics_match = eval_matches[-1]
    payload: Dict[str, object] = {
        "mode": metrics_match.group("mode"),
        "metrics": parse_float_dict(metrics_match),
        "log_path": str(log_path),
    }
    if kind == "factorized":
        diag_matches = list(FACTORIZED_DIAG_RE.finditer(text))
        if not diag_matches:
            raise ValueError(f"Could not parse factorized diagnostics from {log_path}")
        payload["diagnostics"] = parse_float_dict(diag_matches[-1])
    return payload


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(cmd: List[str], env: Dict[str, str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def train_run(
    spec: ExperimentSpec,
    seed: int,
    gpu_group: str,
    run_root: Path,
    epochs: Optional[int],
) -> Path:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_group
    cmd = [
        sys.executable,
        spec.entrypoint,
        "--config",
        spec.config,
        "--seed",
        str(seed),
        "--pl_root_dir",
        str(run_root),
        "--gpus",
        str(len(gpu_group.split(","))),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    train_log = run_root / "suite_logs" / "train_driver.log"
    run_command(cmd, env=env, log_path=train_log)
    return latest_final_ckpt(run_root)


def eval_run(
    spec: ExperimentSpec,
    seed: int,
    gpu_group: str,
    run_root: Path,
    ckpt_path: Path,
    protocol: EvalProtocol,
) -> Dict[str, object]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_group
    cmd = [
        sys.executable,
        spec.entrypoint,
        "--config",
        spec.config,
        "--mode",
        "test",
        "--seed",
        str(seed),
        "--ckpt_path",
        str(ckpt_path),
        "--pl_root_dir",
        str(run_root),
        "--gpus",
        str(len(gpu_group.split(","))),
        "--test_data_path",
        protocol.data_path,
        "--test_with_cand",
        "true" if protocol.test_with_cand else "false",
    ]
    eval_log = run_root / "suite_logs" / f"eval_driver_{protocol.name}.log"
    run_command(cmd, env=env, log_path=eval_log)
    version_dir = ckpt_path.parent
    parsed = parse_eval_log(latest_test_log(version_dir, protocol), kind=spec.kind)
    parsed["protocol"] = protocol.name
    return parsed


def run_job(
    spec: ExperimentSpec,
    seed: int,
    gpu_group: str,
    output_dir: Path,
    epochs: Optional[int],
    skip_train: bool,
    skip_eval: bool,
    force: bool,
) -> Dict[str, object]:
    run_root = output_dir / spec.name / f"seed_{seed}"
    ensure_dir(run_root)
    summary_path = run_root / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if skip_train:
        ckpt_path = latest_final_ckpt(run_root)
    else:
        ckpt_path = train_run(spec, seed, gpu_group, run_root, epochs)

    protocols: Dict[str, object] = {}
    if not skip_eval:
        for protocol in PROTOCOLS:
            protocols[protocol.name] = eval_run(
                spec=spec,
                seed=seed,
                gpu_group=gpu_group,
                run_root=run_root,
                ckpt_path=ckpt_path,
                protocol=protocol,
            )

    summary: Dict[str, object] = {
        "experiment": spec.name,
        "title": spec.title,
        "kind": spec.kind,
        "seed": seed,
        "gpu_group": gpu_group,
        "run_root": str(run_root),
        "checkpoint": str(ckpt_path),
        "protocols": protocols,
    }
    write_json(summary_path, summary)
    return summary


def gather_summaries(output_dir: Path) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for path in sorted(output_dir.glob("*/seed_*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    seq = list(values)
    if not seq:
        return {}
    if len(seq) == 1:
        return {"mean": seq[0], "std": 0.0}
    return {"mean": statistics.mean(seq), "std": statistics.stdev(seq)}


def aggregate_results(summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for item in summaries:
        grouped.setdefault(str(item["experiment"]), []).append(item)

    aggregate: Dict[str, object] = {"experiments": {}, "recommendation": []}
    for exp_name, items in grouped.items():
        exp_payload: Dict[str, object] = {
            "title": str(items[0]["title"]),
            "kind": str(items[0]["kind"]),
            "num_seeds": len(items),
            "protocols": {},
        }
        for protocol in ("r10", "rall"):
            metric_keys = ("r1", "r2", "r5", "r10", "r20", "mrr")
            proto_metrics: Dict[str, Dict[str, float]] = {}
            for key in metric_keys:
                values = [
                    float(run["protocols"][protocol]["metrics"][key])
                    for run in items
                    if protocol in run["protocols"]
                    and key in run["protocols"][protocol]["metrics"]
                ]
                stats = mean_std(values)
                if stats:
                    proto_metrics[key] = stats

            diag_stats: Dict[str, Dict[str, float]] = {}
            for key in (
                "union_size",
                "union_rate",
                "gold_union",
                "gold_style",
                "gold_proto",
                "gold_mmbert",
                "delta_abs",
                "top1_flip",
            ):
                values = [
                    float(run["protocols"][protocol]["diagnostics"][key])
                    for run in items
                    if protocol in run["protocols"]
                    and "diagnostics" in run["protocols"][protocol]
                    and key in run["protocols"][protocol]["diagnostics"]
                ]
                stats = mean_std(values)
                if stats:
                    diag_stats[key] = stats
            exp_payload["protocols"][protocol] = {
                "metrics": proto_metrics,
                "diagnostics": diag_stats,
            }
        aggregate["experiments"][exp_name] = exp_payload

    aggregate["recommendation"] = build_recommendation(aggregate["experiments"])
    return aggregate


def build_recommendation(experiments: Dict[str, object]) -> List[str]:
    def metric(exp_name: str, protocol: str, key: str) -> Optional[float]:
        exp = experiments.get(exp_name)
        if not exp:
            return None
        proto = exp.get("protocols", {}).get(protocol, {})
        stats = proto.get("metrics", {}).get(key, {})
        return stats.get("mean")

    def diag(exp_name: str, protocol: str, key: str) -> Optional[float]:
        exp = experiments.get(exp_name)
        if not exp:
            return None
        proto = exp.get("protocols", {}).get(protocol, {})
        stats = proto.get("diagnostics", {}).get(key, {})
        return stats.get("mean")

    notes: List[str] = []
    base_full = metric("plain_mmbert", "rall", "r1")
    v403_full = metric("v4_03_base_only", "rall", "r1")
    v402_full = metric("v4_02_no_mmbert_branch", "rall", "r1")
    v401_full = metric("v4_01_style_expr_two_stage", "rall", "r1")

    if base_full is not None and v403_full is not None:
        notes.append(
            f"`v4_03_base_only` vs plain MM-BERT full-set r@1 gap: {v403_full - base_full:+.4f}"
        )
    if v402_full is not None and v403_full is not None:
        notes.append(
            f"`v4_02_no_mmbert_branch` vs `v4_03_base_only` full-set r@1 gap: {v402_full - v403_full:+.4f}"
        )
    if v401_full is not None and v403_full is not None:
        notes.append(
            f"`v4_01_style_expr_two_stage` vs `v4_03_base_only` full-set r@1 gap: {v401_full - v403_full:+.4f}"
        )

    top1_flip = diag("v4_01_style_expr_two_stage", "r10", "top1_flip")
    delta_abs = diag("v4_01_style_expr_two_stage", "r10", "delta_abs")
    if top1_flip is not None and delta_abs is not None:
        notes.append(
            "`v4_01_style_expr_two_stage` R10 diagnostics: "
            f"top1_flip={top1_flip:.4f}, delta_abs={delta_abs:.4f}"
        )
    gold_mmbert = diag("v4_01_style_expr_two_stage", "rall", "gold_mmbert")
    if gold_mmbert is not None:
        notes.append(
            f"`v4_01_style_expr_two_stage` full-set gold_in_mmbert_branch_rate={gold_mmbert:.4f}"
        )

    candidates: List[Tuple[str, float]] = []
    for exp_name, exp in experiments.items():
        rall_r1 = exp.get("protocols", {}).get("rall", {}).get("metrics", {}).get("r1", {}).get("mean")
        if rall_r1 is not None:
            candidates.append((exp_name, float(rall_r1)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates:
        notes.append(f"Best full-set mean r@1: `{candidates[0][0]}` = {candidates[0][1]:.4f}")
    return notes


def build_markdown_report(aggregate: Dict[str, object]) -> str:
    exps: Dict[str, object] = aggregate["experiments"]
    lines: List[str] = ["# Core Ablation Summary", ""]
    lines.append("## Full-Set (Rall)")
    lines.append("")
    lines.append("| experiment | seeds | r@1 mean | r@1 std | mrr mean | mrr std |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for exp_name, exp in exps.items():
        rall = exp.get("protocols", {}).get("rall", {}).get("metrics", {})
        r1 = rall.get("r1", {})
        mrr = rall.get("mrr", {})
        lines.append(
            f"| `{exp_name}` | {exp.get('num_seeds', 0)} | "
            f"{r1.get('mean', float('nan')):.4f} | {r1.get('std', float('nan')):.4f} | "
            f"{mrr.get('mean', float('nan')):.4f} | {mrr.get('std', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append("## R10 Candidates")
    lines.append("")
    lines.append("| experiment | r@1 mean | r@1 std | mrr mean | mrr std |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for exp_name, exp in exps.items():
        r10 = exp.get("protocols", {}).get("r10", {}).get("metrics", {})
        r1 = r10.get("r1", {})
        mrr = r10.get("mrr", {})
        lines.append(
            f"| `{exp_name}` | {r1.get('mean', float('nan')):.4f} | {r1.get('std', float('nan')):.4f} | "
            f"{mrr.get('mean', float('nan')):.4f} | {mrr.get('std', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append("## Factorized Diagnostics")
    lines.append("")
    lines.append("| experiment | protocol | gold_union | gold_mmbert | delta_abs | top1_flip |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for exp_name, exp in exps.items():
        for protocol in ("r10", "rall"):
            diag = exp.get("protocols", {}).get(protocol, {}).get("diagnostics", {})
            if not diag:
                continue
            lines.append(
                f"| `{exp_name}` | `{protocol}` | "
                f"{diag.get('gold_union', {}).get('mean', float('nan')):.4f} | "
                f"{diag.get('gold_mmbert', {}).get('mean', float('nan')):.4f} | "
                f"{diag.get('delta_abs', {}).get('mean', float('nan')):.4f} | "
                f"{diag.get('top1_flip', {}).get('mean', float('nan')):.4f} |"
            )
    recs = aggregate.get("recommendation", [])
    if recs:
        lines.append("")
        lines.append("## Recommendation Notes")
        lines.append("")
        for note in recs:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    jobs = [(EXPERIMENTS[name], seed) for seed in args.seeds for name in args.experiments]
    job_groups = split_jobs(jobs, len(args.gpu_groups))

    if not args.skip_train or not args.skip_eval:
        def worker(gpu_group: str, assigned: Sequence[Tuple[ExperimentSpec, int]]) -> None:
            for spec, seed in assigned:
                run_job(
                    spec=spec,
                    seed=seed,
                    gpu_group=gpu_group,
                    output_dir=output_dir,
                    epochs=args.epochs,
                    skip_train=args.skip_train,
                    skip_eval=args.skip_eval,
                    force=args.force,
                )

        with ThreadPoolExecutor(max_workers=len(args.gpu_groups)) as pool:
            futures = []
            for gpu_group, assigned in zip(args.gpu_groups, job_groups):
                if not assigned:
                    continue
                futures.append(pool.submit(worker, gpu_group, assigned))
            for fut in futures:
                fut.result()

    summaries = gather_summaries(output_dir)
    aggregate = aggregate_results(summaries)
    write_json(output_dir / "aggregate.json", aggregate)
    (output_dir / "summary.md").write_text(
        build_markdown_report(aggregate), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "num_summaries": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
