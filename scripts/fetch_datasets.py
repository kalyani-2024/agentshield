"""Download the public source datasets the benchmark is built from.

    python scripts/fetch_datasets.py            # everything, into data/raw/
    python scripts/fetch_datasets.py --only injecagent deepset

Nothing here is generated: every row comes from a published dataset.  Files
already on disk are kept, so a re-run only fetches what is missing.

Sources
-------
deepset      deepset/prompt-injections           Apache-2.0  direct injection + benign prompts
gandalf      Lakera/gandalf_ignore_instructions  MIT         human-written injection attempts
jailbreak    jackhhao/jailbreak-classification   Apache-2.0  jailbreak vs benign prompts
phishing     pirocheto/phishing-url              CC-BY-4.0   phishing vs legitimate URLs
glaive       glaiveai/glaive-function-calling-v2 Apache-2.0  benign tool-calling conversations
injecagent   uiuc-kang-lab/InjecAgent (GitHub)   MIT         indirect injection against tool agents
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_ROWS_API = "https://datasets-server.huggingface.co/rows"
_INJECAGENT = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/"
_PAGE = 100
#: seconds between pages, to stay under the rows API rate limit
_PAUSE = 1.0

#: name -> (hf dataset id, config, [(split, max rows or None)])
HF_SOURCES: dict[str, tuple[str, str, list[tuple[str, int | None]]]] = {
    "deepset": ("deepset/prompt-injections", "default", [("train", None), ("test", None)]),
    "gandalf": (
        "Lakera/gandalf_ignore_instructions",
        "default",
        [("train", None), ("validation", None), ("test", None)],
    ),
    "jailbreak": (
        "jackhhao/jailbreak-classification",
        "default",
        [("train", None), ("test", None)],
    ),
    "phishing": ("pirocheto/phishing-url", "default", [("train", 3000), ("test", 1500)]),
    "glaive": ("glaiveai/glaive-function-calling-v2", "default", [("train", 4000)]),
}

INJECAGENT_FILES = (
    "test_cases_dh_base.json",
    "test_cases_dh_enhanced.json",
    "test_cases_ds_base.json",
    "test_cases_ds_enhanced.json",
    "attacker_simulated_responses.json",
    "user_cases.jsonl",
)


def _get(url: str, retries: int = 8) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "agentshield-fetch/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
            # the rows API rate-limits bursts; back off properly on a 429
            limited = isinstance(error, urllib.error.HTTPError) and error.code == 429
            wait = 30 * (attempt + 1) if limited else 2 ** attempt
            print(f"    retry in {wait}s ({error})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_hf(name: str) -> None:
    dataset, config, splits = HF_SOURCES[name]
    for split, limit in splits:
        target = RAW_DIR / name / f"{split}.jsonl"
        if target.exists():
            print(f"  {name}/{split}: already present")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        offset = 0
        while limit is None or offset < limit:
            query = urllib.parse.urlencode(
                {"dataset": dataset, "config": config, "split": split,
                 "offset": offset, "length": _PAGE}
            )
            page = json.loads(_get(f"{_ROWS_API}?{query}"))
            batch = [item["row"] for item in page.get("rows", [])]
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            time.sleep(_PAUSE)
            if offset >= page.get("num_rows_total", 0):
                break
        if limit is not None:
            rows = rows[:limit]
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name}/{split}: {len(rows)} rows")


def fetch_injecagent() -> None:
    folder = RAW_DIR / "injecagent"
    folder.mkdir(parents=True, exist_ok=True)
    for filename in INJECAGENT_FILES:
        target = folder / filename
        if target.exists():
            print(f"  injecagent/{filename}: already present")
            continue
        target.write_bytes(_get(_INJECAGENT + filename))
        print(f"  injecagent/{filename}: {target.stat().st_size} bytes")


def main() -> None:
    names = [*HF_SOURCES, "injecagent"]
    parser = argparse.ArgumentParser(description="Fetch AgentShield source datasets")
    parser.add_argument("--only", nargs="*", choices=names, default=names)
    args = parser.parse_args()

    print(f"Fetching into {RAW_DIR}")
    for name in args.only:
        if name == "injecagent":
            fetch_injecagent()
        else:
            fetch_hf(name)


if __name__ == "__main__":
    main()
