"""Paired uncertainty and significance statistics for saved model outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def macro_f1(gold: np.ndarray, prediction: np.ndarray, labels: list[str]) -> float:
    scores = []
    for label in labels:
        true_positive = int(np.sum((gold == label) & (prediction == label)))
        false_positive = int(np.sum((gold != label) & (prediction == label)))
        false_negative = int(np.sum((gold == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    numerator = 2 * sum(
        math.comb(discordant, index) for index in range(tail + 1)
    )
    # Keep the exact binomial denominator as an integer.  Converting 2**n to a
    # float first overflows once the paired disagreement count is large.
    probability = numerator / (1 << discordant)
    return min(1.0, probability)


def paired_statistics(
    direct_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    labels: list[str],
    seed: int = 20260820,
    samples: int = 10_000,
) -> dict[str, object]:
    direct = {str(row["case_id"]): row for row in direct_rows}
    candidate = {str(row["case_id"]): row for row in candidate_rows}
    if set(direct) != set(candidate):
        raise ValueError("paired result files have different case identifiers")
    case_ids = sorted(direct)
    gold = np.array([str(direct[key]["gold"]) for key in case_ids], dtype=object)
    if any(str(candidate[key]["gold"]) != expected for key, expected in zip(case_ids, gold, strict=True)):
        raise ValueError("paired result files disagree on gold labels")
    failure = "__failure__"
    left = np.array(
        [str(direct[key]["prediction"]) if direct[key]["prediction"] in labels else failure for key in case_ids],
        dtype=object,
    )
    right = np.array(
        [str(candidate[key]["prediction"]) if candidate[key]["prediction"] in labels else failure for key in case_ids],
        dtype=object,
    )

    def scores(index: np.ndarray) -> tuple[float, float, float, float]:
        y = gold[index]
        a = left[index]
        b = right[index]
        acc_a = float(np.mean(a == y))
        acc_b = float(np.mean(b == y))
        f1_a = macro_f1(y, a, labels)
        f1_b = macro_f1(y, b, labels)
        return acc_a, acc_b, f1_a, f1_b

    full = scores(np.arange(len(case_ids)))
    rng = np.random.default_rng(seed)
    differences = np.empty((samples, 2), dtype=float)
    for iteration in range(samples):
        index = rng.integers(0, len(case_ids), len(case_ids))
        acc_a, acc_b, f1_a, f1_b = scores(index)
        differences[iteration] = (acc_b - acc_a, f1_b - f1_a)
    direct_correct = left == gold
    candidate_correct = right == gold
    wins = int(np.sum(~direct_correct & candidate_correct))
    losses = int(np.sum(direct_correct & ~candidate_correct))
    p_value = exact_mcnemar_p(wins, losses)
    return {
        "records": len(case_ids),
        "seed": seed,
        "bootstrap_samples": samples,
        "direct_accuracy": full[0],
        "candidate_accuracy": full[1],
        "accuracy_difference": full[1] - full[0],
        "accuracy_difference_ci95": np.quantile(differences[:, 0], [0.025, 0.975]).tolist(),
        "direct_macro_f1": full[2],
        "candidate_macro_f1": full[3],
        "macro_f1_difference": full[3] - full[2],
        "macro_f1_difference_ci95": np.quantile(differences[:, 1], [0.025, 0.975]).tolist(),
        "candidate_only_correct": wins,
        "direct_only_correct": losses,
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--labels", default="true,false,both,unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()
    direct = [json.loads(line) for line in args.direct.read_text().splitlines()]
    candidate = [json.loads(line) for line in args.candidate.read_text().splitlines()]
    payload = paired_statistics(
        direct,
        candidate,
        [item.strip() for item in args.labels.split(",") if item.strip()],
        args.seed,
        args.samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
