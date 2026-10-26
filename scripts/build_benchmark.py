"""Assemble the benchmark and the training pools from the fetched datasets.

    python scripts/fetch_datasets.py     # once
    python scripts/build_benchmark.py    # writes data/benchmark and data/train

Outputs
-------
data/benchmark/agentshield_bench_v1.jsonl   500 benign + 500 malicious interactions
data/benchmark/MANIFEST.json                counts per family and per source
data/train/injection_text.jsonl             text pool the injection models are trained on
data/train/urls.jsonl                       URL pool the URL model is trained on

Leakage control
---------------
* Prompt datasets: the benchmark draws from each dataset's published *test*
  split, training from *train* (and *validation*).
* InjecAgent: split by attacker instruction, so no attack instruction that is
  in the benchmark was ever seen in training.
* glaive: the benchmark and the training pool use disjoint conversations.
* Any training text identical to a benchmark text is dropped.

No row is invented.  Where a source does not specify part of an interaction,
that part is left empty or composed from another real source, and the record
says so in ``notes`` (see DESIGN.md, "Benchmark construction").
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BENCH_DIR = ROOT / "data" / "benchmark"
TRAIN_DIR = ROOT / "data" / "train"

SEED = 20261012
BENIGN, MALICIOUS = "benign", "malicious"

#: benchmark quota per family
QUOTA = {
    "direct_injection": 125,
    "indirect_injection": 100,
    "unauthorized_tool": 75,
    "data_exfiltration": 75,
    "multi_step": 75,
    "malicious_url": 50,
    "benign_prompt": 150,
    "benign_tool_call": 125,
    "benign_tool_session": 125,
    "benign_document": 50,
    "benign_url": 50,
}

#: the channel a family's ``text`` arrived on (see detection.strategy.Channel)
CHANNEL = {
    "direct_injection": "user",
    "benign_prompt": "user",
    "benign_tool_call": "user",
    "indirect_injection": "untrusted",
    "unauthorized_tool": "untrusted",
    "data_exfiltration": "untrusted",
    "multi_step": "untrusted",
    "benign_tool_session": "untrusted",
    "benign_document": "untrusted",
    "malicious_url": "url",
    "benign_url": "url",
}
USER, UNTRUSTED = ("user",), ("untrusted",)
BOTH = ("user", "untrusted")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shuffled(items: list, rng: random.Random) -> list:
    out = list(items)
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------
# glaive: real benign tool-calling conversations
# ----------------------------------------------------------------------
_TURN_RE = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):")
_CALL_RE = re.compile(
    r"<functioncall>\s*\{\s*\"name\":\s*\"([^\"]+)\",\s*\"arguments\":\s*'(.*)'\s*\}",
    re.DOTALL,
)
_FUNC_NAME_RE = re.compile(r"\"name\":\s*\"([A-Za-z0-9_.-]+)\"")


def parse_glaive(row: dict) -> dict | None:
    """One conversation -> declared functions plus an ordered list of events."""
    functions = sorted(set(_FUNC_NAME_RE.findall(row["system"])))
    parts = _TURN_RE.split(row["chat"])
    events: list[tuple[str, object]] = []
    for role, body in zip(parts[1::2], parts[2::2]):
        body = body.replace("<|endoftext|>", "").strip()
        if role == "USER":
            events.append(("user", body))
        elif role == "FUNCTION RESPONSE":
            events.append(("response", body))
        else:
            match = _CALL_RE.search(body)
            if not match:
                continue
            try:
                arguments = json.loads(match.group(2))
            except json.JSONDecodeError:
                return None
            if not isinstance(arguments, dict):
                return None
            events.append(("call", (match.group(1), arguments)))
    calls = [e for e in events if e[0] == "call"]
    if not calls or any(name not in functions for _, (name, _a) in calls):
        return None
    return {"functions": functions, "events": events, "n_calls": len(calls)}


def glaive_steps(conversation: dict, max_calls: int = 3) -> list[dict]:
    """Each call becomes a step whose context is what the agent had read so far."""
    steps: list[dict] = []
    context: list[dict] = []
    last_tool = "tool"
    for kind, value in conversation["events"]:
        if kind == "user":
            context.append({"content": value, "source": "user", "trust": "USER"})
        elif kind == "response":
            context.append({"content": value, "source": last_tool, "trust": "UNTRUSTED"})
        else:
            name, arguments = value
            last_tool = name
            steps.append(
                {
                    "framework": "openai",
                    "call": {
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    },
                    "context": list(context),
                    "user_authorized": False,
                }
            )
            if len(steps) == max_calls:
                break
    return steps


# ----------------------------------------------------------------------
# InjecAgent: indirect injection against tool-using agents
# ----------------------------------------------------------------------
def _literal(text: str) -> dict:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return value if isinstance(value, dict) else {}


def _tool_response(case: dict) -> str:
    """The tool output the agent read, with the attacker's text embedded."""
    raw = case["Tool Response"]
    try:
        unquoted = json.loads(raw)
    except json.JSONDecodeError:
        unquoted = raw
    return unquoted if isinstance(unquoted, str) else raw


def _user_step(case: dict) -> dict:
    return {
        "framework": "local",
        "call": {"tool": case["User Tool"], "arguments": _literal(case["Tool Parameters"])},
        "context": [
            {"content": case["User Instruction"], "source": "user", "trust": "USER"}
        ],
        "user_authorized": False,
    }


def _hijack_context(case: dict) -> list[dict]:
    return [
        {"content": case["User Instruction"], "source": "user", "trust": "USER"},
        {"content": _tool_response(case), "source": case["User Tool"], "trust": "UNTRUSTED"},
    ]


def _simulated(sim: dict, tool: str) -> tuple[dict, str] | None:
    """First recorded (parameters, response) pair for an attacker data tool."""
    prefix = f"({tool},"
    for key in sorted(sim):
        if key.startswith(prefix):
            return _literal(key[len(prefix):-1]), sim[key]
    return None


def split_attacker_cases(cases: list[dict], rng: random.Random) -> tuple[set[str], set[str]]:
    """Disjoint attacker-instruction groups for the benchmark and for training."""
    instructions = sorted({c["Attacker Instruction"] for c in cases})
    rng.shuffle(instructions)
    half = len(instructions) // 2
    return set(instructions[:half]), set(instructions[half:])


# ----------------------------------------------------------------------
# sample constructors
# ----------------------------------------------------------------------
def _sample(family: str, label: str, source: str, text: str, steps: list[dict],
            allowed_tools: list[str], notes: str = "") -> dict:
    return {
        "label": label,
        "family": family,
        "source": source,
        "channel": CHANNEL[family],
        "text": text,
        "principal": {"id": "agent-user", "allowed_tools": allowed_tools},
        "steps": steps,
        "notes": notes,
    }


def prompt_sample(family: str, label: str, source: str, text: str, carrier: dict) -> dict:
    """A prompt-only record, attached to a real (benign) carrier tool call.

    The prompt datasets hold user messages, not tool calls.  Both classes get a
    carrier call from the same benign pool, so the message is the only thing
    that differs between a benign and a malicious record.
    """
    name, arguments = next(v for k, v in carrier["events"] if k == "call")
    step = {
        "framework": "gemini",
        "call": {"functionCall": {"name": name, "args": arguments}},
        "context": [{"content": text, "source": "user", "trust": "USER"}],
        "user_authorized": False,
    }
    return _sample(family, label, source, text, [step], carrier["functions"],
                   notes="carrier call from glaive; the prompt is the user message")


def url_sample(family: str, label: str, url: str) -> dict:
    step = {
        "framework": "mcp",
        "call": {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "fetch_url", "arguments": {"url": url}},
        },
        "context": [],
        "user_authorized": False,
    }
    return _sample(family, label, "pirocheto/phishing-url:test", url, [step], ["fetch_url"])


def main() -> None:
    rng = random.Random(SEED)
    bench: list[dict] = []
    train_text: list[dict] = []

    def add_train(text: str, label: str, source: str, channels: tuple[str, ...]) -> None:
        """``channels`` says which per-channel model may learn from this row."""
        if text and text.strip():
            train_text.append({"text": text, "label": label, "source": source,
                               "channels": list(channels)})

    # -- glaive pools ----------------------------------------------------
    conversations = [c for c in map(parse_glaive, _jsonl(RAW / "glaive" / "train.jsonl")) if c]
    conversations = _shuffled(conversations, rng)
    single = [c for c in conversations if c["n_calls"] == 1]
    multi = [c for c in conversations if c["n_calls"] >= 2]

    n_carriers = QUOTA["direct_injection"] + QUOTA["benign_prompt"] + QUOTA["benign_document"]
    carriers, single = single[:n_carriers], single[n_carriers:]
    bench_single, single = single[: QUOTA["benign_tool_call"]], single[QUOTA["benign_tool_call"]:]
    bench_multi, multi = multi[: QUOTA["benign_tool_session"]], multi[QUOTA["benign_tool_session"]:]
    carrier_iter = iter(carriers)

    for conv in bench_single:
        steps = glaive_steps(conv)
        text = steps[0]["context"][-1]["content"] if steps[0]["context"] else ""
        bench.append(_sample("benign_tool_call", BENIGN, "glaive-function-calling-v2",
                             text, steps, conv["functions"]))
    for conv in bench_multi:
        steps = glaive_steps(conv)
        # what a content detector sees: the tool output the agent acted on
        outputs = [c["content"] for c in steps[-1]["context"] if c["trust"] == "UNTRUSTED"]
        text = outputs[-1] if outputs else steps[-1]["context"][-1]["content"]
        bench.append(_sample("benign_tool_session", BENIGN, "glaive-function-calling-v2",
                             text, steps, conv["functions"]))

    # the rest of glaive is benign training text: prompts, arguments, tool output
    for conv in (single + multi)[:700]:
        for kind, value in conv["events"]:
            if kind == "user":
                add_train(value, BENIGN, "glaive:user", USER)
            elif kind == "response":
                add_train(value, BENIGN, "glaive:tool_output", UNTRUSTED)
            else:
                add_train("\n".join(f"{k}\n{v}" for k, v in value[1].items()),
                          BENIGN, "glaive:arguments", USER)

    # -- prompt datasets -------------------------------------------------
    deepset_test = _jsonl(RAW / "deepset" / "test.jsonl")
    deepset_train = _jsonl(RAW / "deepset" / "train.jsonl")
    jb_test = _jsonl(RAW / "jailbreak" / "test.jsonl")
    jb_train = _jsonl(RAW / "jailbreak" / "train.jsonl")
    gandalf_test = _jsonl(RAW / "gandalf" / "test.jsonl")
    gandalf_train = (_jsonl(RAW / "gandalf" / "train.jsonl")
                     + _jsonl(RAW / "gandalf" / "validation.jsonl"))

    direct = (
        [("deepset/prompt-injections:test", r["text"])
         for r in _shuffled([r for r in deepset_test if r["label"] == 1], rng)[:50]]
        + [("Lakera/gandalf_ignore_instructions:test", r["text"])
           for r in _shuffled(gandalf_test, rng)[:40]]
        + [("jackhhao/jailbreak-classification:test", r["prompt"])
           for r in _shuffled([r for r in jb_test if r["type"] == "jailbreak"], rng)[:35]]
    )
    for source, text in direct:
        bench.append(prompt_sample("direct_injection", MALICIOUS, source, text,
                                   next(carrier_iter)))

    deepset_benign_test = _shuffled([r for r in deepset_test if r["label"] == 0], rng)
    benign_prompts = (
        [("deepset/prompt-injections:test", r["text"]) for r in deepset_benign_test]
        + [("jackhhao/jailbreak-classification:test", r["prompt"])
           for r in _shuffled([r for r in jb_test if r["type"] == "benign"], rng)]
    )[: QUOTA["benign_prompt"]]
    for source, text in benign_prompts:
        bench.append(prompt_sample("benign_prompt", BENIGN, source, text, next(carrier_iter)))

    # An override attempt is an attack on either channel, so those rows teach
    # both models.  jailbreak's benign rows are role-play prompts ("act as a
    # yoga instructor"): fine from a user, but an instruction if a document
    # says it - so they teach the user model only.
    for row in deepset_train:
        add_train(row["text"], MALICIOUS if row["label"] == 1 else BENIGN,
                  "deepset:train", BOTH)
    for row in jb_train:
        malicious = row["type"] == "jailbreak"
        add_train(row["prompt"], MALICIOUS if malicious else BENIGN,
                  "jailbreak:train", BOTH if malicious else USER)
    for row in gandalf_train:
        add_train(row["text"], MALICIOUS, "gandalf:train", BOTH)

    # -- InjecAgent ------------------------------------------------------
    inj = RAW / "injecagent"
    dh = {"base": _json(inj / "test_cases_dh_base.json"),
          "enhanced": _json(inj / "test_cases_dh_enhanced.json")}
    ds = {"base": _json(inj / "test_cases_ds_base.json"),
          "enhanced": _json(inj / "test_cases_ds_enhanced.json")}
    sim = _json(inj / "attacker_simulated_responses.json")

    dh_test, dh_train = split_attacker_cases(dh["base"], rng)
    ds_test, ds_train = split_attacker_cases(ds["base"], rng)

    def pool(cases: dict, keep: set[str]) -> list[tuple[str, dict]]:
        """Base and enhanced variants, interleaved, restricted to one group."""
        out = [(variant, case) for variant, rows in cases.items() for case in rows
               if case["Attacker Instruction"] in keep]
        return _shuffled(out, rng)

    dh_bench = pool(dh, dh_test)
    n_ind, n_unauth = QUOTA["indirect_injection"], QUOTA["unauthorized_tool"]
    for variant, case in dh_bench[:n_ind]:
        hijack = {
            "framework": "local",
            "call": {"tool": case["Attacker Tools"][0], "arguments": {}},
            "context": _hijack_context(case),
            "user_authorized": False,
        }
        bench.append(_sample(
            "indirect_injection", MALICIOUS, f"InjecAgent:dh_{variant}",
            _tool_response(case), [_user_step(case), hijack], ["*"],
            notes="InjecAgent names the hijacked tool but not its arguments; left empty",
        ))
    for variant, case in dh_bench[n_ind:n_ind + n_unauth]:
        hijack = {
            "framework": "local",
            "call": {"tool": case["Attacker Tools"][0], "arguments": {}},
            "context": _hijack_context(case),
            "user_authorized": False,
        }
        bench.append(_sample(
            "unauthorized_tool", MALICIOUS, f"InjecAgent:dh_{variant}",
            _tool_response(case), [_user_step(case), hijack], [case["User Tool"]],
            notes="least-privilege principal: only the tool the user's task needs is granted",
        ))

    ds_bench = [
        (variant, case) for variant, case in pool(ds, ds_test)
        if _simulated(sim, case["Attacker Tools"][0])
        and _EMAIL_RE.search(case["Attacker Instruction"])
    ]
    n_exfil, n_multi = QUOTA["data_exfiltration"], QUOTA["multi_step"]

    def exfil_steps(case: dict) -> tuple[dict, dict]:
        read_tool = case["Attacker Tools"][0]
        params, stolen = _simulated(sim, read_tool)
        attacker = _EMAIL_RE.search(case["Attacker Instruction"]).group(0)
        read = {
            "framework": "local",
            "call": {"tool": read_tool, "arguments": params},
            "context": _hijack_context(case),
            "user_authorized": False,
        }
        send = {
            "framework": "local",
            "call": {"tool": "GmailSendEmail", "arguments": {"to": attacker, "body": stolen}},
            "context": _hijack_context(case)
            + [{"content": stolen, "source": read_tool, "trust": "UNTRUSTED"}],
            "user_authorized": False,
        }
        return read, send

    for variant, case in ds_bench[:n_exfil]:
        _read, send = exfil_steps(case)
        bench.append(_sample(
            "data_exfiltration", MALICIOUS, f"InjecAgent:ds_{variant}",
            _tool_response(case), [send], ["*"],
            notes="final exfiltration call only, in a fresh session; body is InjecAgent's "
                  "simulated response of the data tool",
        ))
    for variant, case in ds_bench[n_exfil:n_exfil + n_multi]:
        read, send = exfil_steps(case)
        bench.append(_sample(
            "multi_step", MALICIOUS, f"InjecAgent:ds_{variant}",
            _tool_response(case), [_user_step(case), read, send], ["*"],
            notes="full chain in one session: user's tool, attacker's data read, exfiltration",
        ))

    for variant, case in pool(dh, dh_train)[:250] + pool(ds, ds_train)[:250]:
        add_train(_tool_response(case), MALICIOUS, f"injecagent:{variant}", UNTRUSTED)

    # benign documents: the same InjecAgent tool-output carriers, holding a real
    # benign text instead of an attack.  Without these a detector could "win" by
    # recognising the carrier rather than the injection.
    user_cases = _jsonl(inj / "user_cases.jsonl")
    benign_payloads = _shuffled(
        [r["text"] for r in deepset_train if r["label"] == 0 and len(r["text"]) < 400], rng
    )
    bench_payloads = [r["text"] for r in deepset_benign_test if len(r["text"]) < 400]

    def benign_document(case: dict, payload: str) -> str:
        return case["Tool Response Template"].replace("<Attacker Instruction>", payload)

    for index in range(QUOTA["benign_document"]):
        case = user_cases[index % len(user_cases)]
        payload = bench_payloads[index % len(bench_payloads)]
        document = benign_document(case, payload)
        carrier = next(carrier_iter)
        name, arguments = next(v for k, v in carrier["events"] if k == "call")
        follow_up = {
            "framework": "local",
            "call": {"tool": name, "arguments": arguments},
            "context": [
                {"content": case["User Instruction"], "source": "user", "trust": "USER"},
                {"content": document, "source": case["User Tool"], "trust": "UNTRUSTED"},
            ],
            "user_authorized": False,
        }
        bench.append(_sample(
            "benign_document", BENIGN, "InjecAgent:user_cases+deepset:test",
            document, [_user_step(case), follow_up], ["*"],
            notes="InjecAgent tool-output carrier holding a benign deepset text; "
                  "follow-up call from glaive",
        ))
    for index, payload in enumerate(benign_payloads[:340]):
        add_train(benign_document(user_cases[index % len(user_cases)], payload),
                  BENIGN, "injecagent:benign_document", UNTRUSTED)

    # -- URLs ------------------------------------------------------------
    url_test = _shuffled(_jsonl(RAW / "phishing" / "test.jsonl"), rng)
    bad = [r["url"] for r in url_test if r["status"] == "phishing"][: QUOTA["malicious_url"]]
    good = [r["url"] for r in url_test if r["status"] == "legitimate"][: QUOTA["benign_url"]]
    bench.extend(url_sample("malicious_url", MALICIOUS, url) for url in bad)
    bench.extend(url_sample("benign_url", BENIGN, url) for url in good)
    bench_urls = set(bad) | set(good)
    train_urls = [
        {"url": r["url"], "label": MALICIOUS if r["status"] == "phishing" else BENIGN}
        for r in _jsonl(RAW / "phishing" / "train.jsonl")
        if r["url"] not in bench_urls
    ]

    # -- finalise --------------------------------------------------------
    counts = Counter(s["family"] for s in bench)
    short = {f: (counts[f], q) for f, q in QUOTA.items() if counts[f] != q}
    if short:
        raise SystemExit(f"quota not met (have, want): {short}")

    bench_texts = {_norm(s["text"]) for s in bench}
    seen: set[str] = set()
    clean_train = []
    for row in train_text:
        key = _norm(row["text"])
        if key in bench_texts or key in seen:
            continue
        seen.add(key)
        clean_train.append(row)

    bench.sort(key=lambda s: _stable_key(s["family"] + s["text"] + json.dumps(s["steps"])))
    for index, sample in enumerate(bench):
        sample["id"] = f"as-{index:04d}"

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write(BENCH_DIR / "agentshield_bench_v1.jsonl", bench)
    write(TRAIN_DIR / "injection_text.jsonl", clean_train)
    write(TRAIN_DIR / "urls.jsonl", train_urls)

    manifest = {
        "name": "agentshield_bench_v1",
        "seed": SEED,
        "total": len(bench),
        "labels": dict(Counter(s["label"] for s in bench)),
        "families": dict(sorted(counts.items())),
        "sources": dict(sorted(Counter(s["source"] for s in bench).items())),
        "train_text": dict(sorted(Counter(
            f"{r['label']}|{r['source']}" for r in clean_train).items())),
        "train_urls": dict(Counter(r["label"] for r in train_urls)),
    }
    (BENCH_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
