"""Summarize evaluation manifests and plot score per round by variant."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path(__file__).with_name("logs") / "evaluations"
DISPLAY_LABELS = {
    "model-a-safe-mask-blast-through-crates-v2": "v2 blast fix",
    "model-a-time-aware-survival-v3": "v3 strict gate",
    "model-a-time-aware-survival-v3-balanced": "v3 balanced",
    "model-a-v3-balanced-mixed-opponents": "v3 balanced mixed",
    "model-a-v4-offense-features-300-mixed": "v4 offense mixed",
    "model-a-v4-offense-features-300-rule": "v4 offense rule",
}


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "evaluation" or payload.get("status") != "completed":
            continue
        evaluation = payload["evaluation"]
        metrics = payload.get("target_metrics")
        if not metrics:
            continue
        rows.append({
            "run_id": payload["run_id"],
            "variant": payload["variant"],
            "scenario": evaluation["scenario"],
            "opponents": ",".join(evaluation["agents"][1:]),
            "seed": evaluation["seed"],
            "rounds": metrics["rounds"],
            "score_per_round": metrics["score_per_round"],
            "coins": metrics["coins"],
            "kills": metrics["kills"],
            "suicides": metrics["suicides"],
            "invalid_actions": metrics["invalid_actions"],
            "mean_decision_time_ms": metrics["mean_decision_time_ms"],
            "checkpoint_sha256": (payload.get("checkpoint") or {}).get("sha256", ""),
        })
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["scenario"], row["opponents"])].append(row)
    summaries = []
    for (variant, scenario, opponents), group in sorted(grouped.items()):
        scores = [float(row["score_per_round"]) for row in group]
        suicides_per_100 = [100.0 * int(row["suicides"]) / int(row["rounds"]) for row in group]
        summaries.append({
            "variant": variant,
            "scenario": scenario,
            "opponents": opponents,
            "runs": len(group),
            "rounds": sum(int(row["rounds"]) for row in group),
            "score_per_round_mean": statistics.mean(scores),
            "score_per_round_sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "coins_per_100_rounds": 100.0 * sum(int(row["coins"]) for row in group) / sum(int(row["rounds"]) for row in group),
            "kills_per_100_rounds": 100.0 * sum(int(row["kills"]) for row in group) / sum(int(row["rounds"]) for row in group),
            "suicides_per_100_rounds_mean": statistics.mean(suicides_per_100),
            "suicides_per_100_rounds_sd": statistics.stdev(suicides_per_100) if len(suicides_per_100) > 1 else 0.0,
            "invalid_actions_per_100_rounds": 100.0 * sum(int(row["invalid_actions"]) for row in group) / sum(int(row["rounds"]) for row in group),
            "mean_decision_time_ms": statistics.mean(float(row["mean_decision_time_ms"]) for row in group),
        })
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No completed evaluation manifests found")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_summaries(path: Path, summaries: list[dict], rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; run this inside the project Docker image.") from exc
    labels = [
        f'{textwrap.fill(DISPLAY_LABELS.get(row["variant"], row["variant"]), width=18)}\n(n={row["runs"]})'
        for row in summaries
    ]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(max(11, 2.7 * len(labels)), 5.2))

    score_means = [row["score_per_round_mean"] for row in summaries]
    score_errors = [row["score_per_round_sd"] for row in summaries]
    axes[0].bar(positions, score_means, yerr=score_errors, capsize=4, color="#2878B5")
    axes[0].axhline(2.3, color="#D95319", linestyle="--", linewidth=1.5, label="acceptance: 2.3")
    axes[0].set_ylabel("Score per round")
    axes[0].set_title("Score (mean ± sample SD)")
    axes[0].legend(frameon=False)

    suicide_means = [row["suicides_per_100_rounds_mean"] for row in summaries]
    suicide_errors = [row["suicides_per_100_rounds_sd"] for row in summaries]
    axes[1].bar(positions, suicide_means, yerr=suicide_errors, capsize=4, color="#E07A5F")
    axes[1].axhline(30.0, color="#333333", linestyle="--", linewidth=1.5, label="acceptance: <30")
    axes[1].set_ylabel("Suicides per 100 rounds")
    axes[1].set_title("Safety (mean ± sample SD)")
    axes[1].legend(frameon=False)

    for index, summary in enumerate(summaries):
        matching = [
            row for row in rows
            if row["variant"] == summary["variant"]
            and row["scenario"] == summary["scenario"]
            and row["opponents"] == summary["opponents"]
        ]
        count = len(matching)
        offsets = [0.0] if count == 1 else [(-0.12 + 0.24 * item / (count - 1)) for item in range(count)]
        axes[0].scatter(
            [index + offset for offset in offsets],
            [row["score_per_round"] for row in matching],
            color="white", edgecolor="#174A70", zorder=3,
        )
        axes[1].scatter(
            [index + offset for offset in offsets],
            [100.0 * row["suicides"] / row["rounds"] for row in matching],
            color="white", edgecolor="#8F3F2D", zorder=3,
        )

    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", labelrotation=0, labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Model A evaluation by opponent mix")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("experiments/plots/evaluation_summary.csv"))
    parser.add_argument("--plot", type=Path, default=Path("experiments/plots/evaluation_score.png"))
    args = parser.parse_args(argv)
    paths = args.manifests or sorted(LOG_DIR.glob("*.json"))
    paths = [path for path in paths if not path.name.endswith(".stats.json")]
    rows = load_rows(paths)
    summaries = summarize_rows(rows)
    write_csv(args.csv, summaries)
    plot_summaries(args.plot, summaries, rows)
    print(f"Wrote {args.csv} and {args.plot}")


if __name__ == "__main__":
    main()
