"""Train the three shipped models from the training pools.

    pip install -e ".[train]"
    python scripts/train_models.py

Reads  data/train/injection_text.jsonl and data/train/urls.jsonl
Writes agentshield/models/injection_user_lr.json, injection_untrusted_lr.json
       and url_lr.json

The benchmark file is never opened here; it stays held out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentshield.detection.ml import (  # noqa: E402
    INJECTION_MODELS,
    MODEL_DIR,
    URL_MODEL,
    LinearTextModel,
)

TRAIN_DIR = ROOT / "data" / "train"


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    text_rows = _jsonl(TRAIN_DIR / "injection_text.jsonl")
    for channel, filename in INJECTION_MODELS.items():
        rows = [row for row in text_rows if channel.value in row["channels"]]
        model = LinearTextModel.fit(
            [row["text"] for row in rows],
            [row["label"] == "malicious" for row in rows],
            analyzer="text",
            c=10.0,
            min_df=2,
            max_features=120_000,
            meta={"trained_on": "data/train/injection_text.jsonl", "channel": channel.value},
        )
        model.save(MODEL_DIR / filename)
        bad = sum(row["label"] == "malicious" for row in rows)
        print(f"{channel.value} injection model: {len(rows)} rows ({bad} malicious), "
              f"{len(model.vocabulary)} features")

    url_rows = _jsonl(TRAIN_DIR / "urls.jsonl")
    urls = LinearTextModel.fit(
        [row["url"] for row in url_rows],
        [row["label"] == "malicious" for row in url_rows],
        analyzer="url",
        c=10.0,
        min_df=2,
        max_features=80_000,
        meta={"trained_on": "data/train/urls.jsonl"},
    )
    urls.save(MODEL_DIR / URL_MODEL)
    print(f"url model: {len(url_rows)} rows, {len(urls.vocabulary)} features")

    for name in (*INJECTION_MODELS.values(), URL_MODEL):
        size = (MODEL_DIR / name).stat().st_size / 1e6
        print(f"  {MODEL_DIR / name}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
