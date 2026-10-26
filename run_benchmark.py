"""The experimental benchmark: four detection approaches on real data.

    python run_benchmark.py                    # rules, ML, hybrid (+ LLM if a key is set)
    python run_benchmark.py --limit 200        # a quick subset
    python run_benchmark.py --no-llm           # skip the LLM arms even if a key is set

The LLM-judge and the true hybrid arm need a provider key in the environment
(GROQ_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY).  Verdicts are cached in
data/cache/, so a re-run costs nothing and gives the same numbers.

Results are printed and written to results/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentshield.benchmark import (
    NaiveBayesClassifier,
    Report,
    evaluate_detector,
    evaluate_engine,
    evaluate_url_model,
    load_benchmark,
    load_training_text,
)
from agentshield.config import load_dotenv
from agentshield.detection import (
    Channel,
    EnsembleStrategy,
    FallbackStrategy,
    HybridStrategy,
    JudgeCache,
    MLStrategy,
    QuotaExhausted,
    build_judge,
    injection_models,
    url_model,
)

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "benchmark_results.json"
CACHE_DIR = ROOT / "data" / "cache"

_METRICS = ("precision", "recall", "f1", "fpr", "attack_success_rate")
_HEADS = ("precision", "recall", "F1", "FPR", "ASR")


def _table(reports: list[Report]) -> None:
    header = f"{'approach':<16}" + "".join(f"{h:>11}" for h in _HEADS) \
        + f"{'avg ms':>10}{'p95 ms':>10}"
    print(header)
    print("-" * len(header))
    for report in reports:
        row = report.as_dict()
        print(f"{report.name:<16}" + "".join(f"{row[m]:>11.3f}" for m in _METRICS)
              + f"{report.avg_latency_ms:>10.2f}{report.p95_latency_ms:>10.2f}")


def _family_table(reports: list[Report], malicious: bool) -> None:
    families = [
        family for family, matrix in sorted(reports[0].per_family.items())
        if (matrix.tp + matrix.fn > 0) == malicious
    ]
    kind = "attacks stopped" if malicious else "benign interactions wrongly stopped"
    print(f"\n  {kind}, per family (count / total)")
    header = f"  {'family':<22}" + "".join(f"{r.name:>14}" for r in reports)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for family in families:
        cells = []
        for report in reports:
            rate = report.family_rates().get(family, {"flagged": 0, "total": 0})
            cells.append(f"{rate['flagged']}/{rate['total']}")
        print(f"  {family:<22}" + "".join(f"{cell:>14}" for cell in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentShield benchmark")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N samples")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM-judge arms")
    parser.add_argument("--no-llm-engine", action="store_true",
                        help="skip the end-to-end llm_judge arm, the one that needs most calls")
    parser.add_argument("--provider", default="", help="groq | gemini | openai")
    parser.add_argument("--model", default="", help="override the provider's default model")
    parser.add_argument("--low", type=float, default=0.15)
    parser.add_argument("--high", type=float, default=0.85)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    samples = load_benchmark()
    if args.limit:
        samples = samples[: args.limit]
    text_samples = [s for s in samples if not s.is_url]
    url_samples = [s for s in samples if s.is_url]
    n_bad = sum(s.is_malicious for s in samples)

    print("=" * 78)
    print("AGENTSHIELD BENCHMARK".center(78))
    print("=" * 78)
    print(f"\nDataset : {len(samples)} interactions ({len(samples) - n_bad} benign, "
          f"{n_bad} malicious), held out from training")

    models = injection_models()
    urls = url_model()
    if models is None or urls is None:
        raise SystemExit("models missing - run: python scripts/train_models.py")

    rules = EnsembleStrategy()
    ml = MLStrategy(models)

    judge = None
    if not args.no_llm:
        # patient retries: a benchmark run should wait out per-minute limits
        judge = build_judge(args.provider, model=args.model, timeout=60, retries=8)
    if judge is not None:
        label = judge.client.label
        judge.cache = JudgeCache(CACHE_DIR / f"judge_{label.replace(':', '_').replace('/', '_')}.jsonl")
        print(f"LLM judge: {label}  ({len(judge.cache)} cached verdicts)")
    else:
        print("LLM judge: NOT RUN - no provider key in the environment. The llm_judge arm is"
              "\n           skipped and 'hybrid' runs as ML alone (nothing to escalate to).")

    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}

    def hybrid() -> HybridStrategy:
        return HybridStrategy(ml, judge, args.low, args.high)

    def judge_ran_out(arm: str) -> bool:
        """After an arm that used the judge: did the daily quota end mid-run?

        The hybrid cascade keeps the model's verdict when the judge cannot
        answer, which is right in production but would pass ML numbers off as
        hybrid ones here, so such an arm is discarded and reported as failed.
        """
        if judge is not None and judge.quota_exhausted:
            failed[arm] = "LLM provider daily quota exhausted mid-run; re-run after it resets"
            print(f"  {arm}: {failed[arm]}")
            return True
        return False

    # ------------------------------------------------------------------
    # LLM-free arms first, then the LLM arms in priority order: the hybrid
    # (the deployed configuration) before the pure judge, so a provider quota
    # that runs out costs the least important numbers.  Verdicts are cached,
    # so a later run resumes where this one stopped.
    print("\nRunning detectors...")
    bayes = MLStrategy(
        {channel: NaiveBayesClassifier().fit(*load_training_text(channel))
         for channel in (Channel.USER, Channel.UNTRUSTED)},
        "naive_bayes",
    )
    detectors: dict[str, Report] = {
        "rules": evaluate_detector(rules, text_samples, "rules"),
        "naive_bayes": evaluate_detector(bayes, text_samples, "naive_bayes"),
        "ml": evaluate_detector(ml, text_samples, "ml"),
    }
    engines: dict[str, Report] = {
        # the all-rules baseline: signatures for content, lexical hints for URLs
        "rules": evaluate_engine(samples, rules, None, "rules"),
        "ml": evaluate_engine(samples, ml, urls, "ml"),
    }
    url_report = evaluate_url_model(urls, url_samples)

    hybrid_stats: dict = {}
    if judge is not None and not judge.quota_exhausted:
        print("  hybrid (detector level)...", flush=True)
        cascade = hybrid()
        report = evaluate_detector(cascade, text_samples, "hybrid", progress=True)
        report.extra.update(escalated_to_llm=cascade.escalated,
                            escalation_rate=round(cascade.escalation_rate, 4),
                            judge_failures=cascade.judge_failures)
        if not judge_ran_out("hybrid"):
            detectors["hybrid"] = report
            hybrid_stats = report.extra

    if judge is not None and not judge.quota_exhausted:
        print("  hybrid (end to end)...", flush=True)
        cascade = hybrid()
        report = evaluate_engine(samples, cascade, urls, "hybrid", progress=True)
        report.extra.update(escalated_to_llm=cascade.escalated,
                            escalation_rate=round(cascade.escalation_rate, 4),
                            judge_failures=cascade.judge_failures)
        if not judge_ran_out("hybrid_engine"):
            engines["hybrid"] = report

    if judge is not None and not judge.quota_exhausted:
        print("  llm_judge (detector level)...", flush=True)
        try:
            detectors["llm_judge"] = evaluate_detector(judge, text_samples, "llm_judge",
                                                       progress=True)
        except QuotaExhausted:
            judge_ran_out("llm_judge")

    if judge is not None and args.no_llm_engine:
        skipped["llm_judge_engine"] = "skipped with --no-llm-engine"
    elif judge is not None and not judge.quota_exhausted:
        print("  llm_judge (end to end)...", flush=True)
        # the deployed "llm" mode: the judge first, the model if it cannot answer
        errors_before = judge.errors
        chain = FallbackStrategy(judge, FallbackStrategy(ml, rules))
        report = evaluate_engine(samples, chain, urls, "llm_judge", progress=True)
        report.extra["fell_back_to_ml"] = judge.errors - errors_before
        if not judge_ran_out("llm_judge_engine"):
            engines["llm_judge"] = report

    if judge is None:
        # without a judge the cascade is the model alone; report it as such
        detectors["hybrid"] = evaluate_detector(hybrid(), text_samples, "hybrid")
        engines["hybrid"] = evaluate_engine(samples, hybrid(), urls, "hybrid")

    order = ("rules", "naive_bayes", "ml", "llm_judge", "hybrid")
    detector_reports = [detectors[k] for k in order if k in detectors]
    engine_reports = [engines[k] for k in order if k in engines]

    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"1. DETECTOR COMPARISON - content only ({len(text_samples)} text samples)")
    print("-" * 78)
    _table(detector_reports)
    if hybrid_stats:
        print(f"\n  hybrid sent {hybrid_stats['escalated_to_llm']} of {len(text_samples)} texts "
              f"({hybrid_stats['escalation_rate']:.1%}) to the LLM judge")
    for report in detector_reports:
        if report.extra.get("unanswered"):
            print(f"  {report.name}: {report.extra['unanswered']} texts got no verdict "
                  f"(counted as not detected)")
    _family_table(detector_reports, malicious=True)
    _family_table(detector_reports, malicious=False)

    print(f"\n  URL model on {len(url_samples)} URLs: precision {url_report.matrix.precision:.3f}"
          f"  recall {url_report.matrix.recall:.3f}  F1 {url_report.matrix.f1:.3f}"
          f"  FPR {url_report.matrix.false_positive_rate:.3f}")

    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"2. END-TO-END ENGINE - full pipeline ({len(samples)} interactions)")
    print("-" * 78)
    _table(engine_reports)
    _family_table(engine_reports, malicious=True)
    _family_table(engine_reports, malicious=False)

    print("\n  how stopped interactions were handled")
    for report in engine_reports:
        d = report.decisions
        print(f"  {report.name:<12} attacks: {d['attack_BLOCK']} blocked, "
              f"{d['attack_REQUIRE_APPROVAL']} held for approval   |   benign: "
              f"{d['benign_BLOCK']} blocked, {d['benign_REQUIRE_APPROVAL']} held for approval")
    for arm, reason in {**failed, **skipped}.items():
        print(f"  NOT IN THE TABLES: {arm} - {reason}")
    if judge is not None:
        print(f"\n  LLM calls this run: {judge.calls} (cache hits {judge.cache_hits}, "
              f"errors {judge.errors})")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "dataset": {"total": len(samples), "malicious": n_bad,
                            "benign": len(samples) - n_bad},
                "llm_judge": judge.client.label if judge is not None else None,
                "hybrid_bounds": {"low": args.low, "high": args.high},
                "detector_level": [r.as_dict() for r in detector_reports],
                "url_model": url_report.as_dict(),
                "end_to_end": [r.as_dict() for r in engine_reports],
                "failed_arms": failed,
                "skipped_arms": skipped,
                "llm_calls": None if judge is None else {
                    "calls": judge.calls, "cache_hits": judge.cache_hits,
                    "errors": judge.errors},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults written to {RESULTS_PATH.relative_to(ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
