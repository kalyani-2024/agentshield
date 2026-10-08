"""Stage 3: the experimental benchmark, printed to the terminal.

    python run_benchmark.py                 # 500 benign + 500 malicious
    python run_benchmark.py --benign 200 --malicious 200 --seed 42

It builds a labelled dataset, trains the ML detector on the training split, then
compares four detection strategies - Rules, Heuristic, ML, Hybrid - on the held
-out test set, and finally runs the whole AgentShield engine end to end.
"""

from __future__ import annotations

import argparse

from agentshield.benchmark import (
    HybridStrategy,
    MLStrategy,
    NaiveBayesClassifier,
    compare_detectors,
    evaluate_engine,
    generate_dataset,
    train_test_split,
)
from agentshield.benchmark.harness import StrategyReport
from agentshield.detection.strategies import HeuristicStrategy, RegexStrategy

_COLUMNS = ("precision", "recall", "f1", "fpr", "accuracy")


def _print_detector_table(reports: dict[str, StrategyReport]) -> None:
    header = f"{'strategy':<12}" + "".join(f"{c:>11}" for c in _COLUMNS) + f"{'latency_ms':>13}"
    print(header)
    print("-" * len(header))
    for name, report in reports.items():
        m = report.matrix.as_dict()
        row = f"{name:<12}" + "".join(f"{m[c]:>11.3f}" for c in _COLUMNS)
        row += f"{report.avg_latency_ms:>13.4f}"
        print(row)


def _print_confusion(reports: dict[str, StrategyReport]) -> None:
    print(f"\n{'strategy':<12}{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}"
          f"{'attack_success':>16}")
    print("-" * 52)
    for name, report in reports.items():
        m = report.matrix
        print(f"{name:<12}{m.tp:>6}{m.fp:>6}{m.tn:>6}{m.fn:>6}"
              f"{m.attack_success_rate:>16.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentShield Stage 3 benchmark")
    parser.add_argument("--benign", type=int, default=500)
    parser.add_argument("--malicious", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 74)
    print("  AGENTSHIELD STAGE 3 BENCHMARK".center(74))
    print("=" * 74)

    dataset = generate_dataset(args.benign, args.malicious, seed=args.seed)
    train, test = train_test_split(dataset, test_ratio=0.3, seed=args.seed)
    n_mal = sum(1 for s in dataset if s.is_malicious)
    print(f"\nDataset : {len(dataset)} interactions "
          f"({len(dataset) - n_mal} benign, {n_mal} malicious)")
    print(f"Split   : {len(train)} train / {len(test)} test (stratified)")

    # train the ML arm on the training split only
    clf = NaiveBayesClassifier()
    clf.fit([s.text for s in train], [s.label for s in train])

    rules = RegexStrategy()
    ml = MLStrategy(clf)
    strategies = {
        "rules": rules,
        "heuristic": HeuristicStrategy(),
        "ml": ml,
        "hybrid": HybridStrategy(rules, ml),
    }

    print("\n" + "-" * 74)
    print("  DETECTOR COMPARISON  (held-out test set, injection text only)")
    print("-" * 74)
    reports = compare_detectors(strategies, test, threshold=args.threshold)
    _print_detector_table(reports)
    _print_confusion(reports)

    print("\n" + "-" * 74)
    print("  END-TO-END ENGINE  (full pipeline over the whole dataset)")
    print("-" * 74)
    engine_report, per_type = evaluate_engine(dataset)
    em = engine_report.matrix
    print(f"\nOverall: detection rate (recall) {em.recall:.3f} | "
          f"false-positive rate {em.false_positive_rate:.3f} | "
          f"attack success {em.attack_success_rate:.3f}")
    print(f"Added latency per call: {engine_report.avg_latency_ms:.4f} ms "
          f"(avg over {engine_report.samples} calls)")

    print(f"\n  {'attack type':<22}{'caught':>8}{'total':>8}{'success_rate':>14}")
    print("  " + "-" * 50)
    for attack_type, matrix in sorted(per_type.items()):
        if attack_type == "benign":
            continue
        total = matrix.tp + matrix.fn
        print(f"  {attack_type:<22}{matrix.tp:>8}{total:>8}"
              f"{matrix.attack_success_rate:>14.3f}")
    benign = per_type.get("benign")
    if benign:
        print(f"\n  benign calls          : {benign.tn} allowed, "
              f"{benign.fp} false-flagged (FPR {benign.false_positive_rate:.3f})")

    print("\n" + "=" * 74)
    print("  Takeaways")
    print("  - Rules: fast and zero false positives, but ~0.5 recall - evasive")
    print("    phrasings dodge the signatures (the real coverage gap).")
    print("  - ML/Hybrid score near-perfect on this SYNTHETIC set because the")
    print("    templates make the classes linearly separable; treat those as an")
    print("    upper bound, not a field result.")
    print("  - End-to-end engine is the headline: it catches 5 of 6 attack")
    print("    families outright; only stealth exfiltration (no injection words,")
    print("    obfuscated secret, neutral destination) slips - which is exactly")
    print("    the case a text detector cannot see and defence-in-depth is for.")
    print("  - An LLM-judge arm plugs in via LLMJudgeStrategy when an API is set.")
    print("=" * 74)


if __name__ == "__main__":
    main()
