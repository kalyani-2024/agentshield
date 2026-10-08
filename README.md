# AgentShield

A runtime security firewall for tool-using AI agents.

An LLM chatbot that reads a poisoned document produces bad text. An *agent* that
reads the same document can turn those injected instructions into real tool
calls: read the credentials file, look up an external contact, send the mail.
AgentShield sits between the agent and its tools and decides, per call, whether
the action is safe.

```
LLM Agent  ->  AGENTSHIELD  ->  actual tools / APIs
```

This repository is the complete project:

- **Stage 1 - core security engine**: adapters, the filter pipeline, risk
  scoring, policy decisions, session privilege states, the audit trail and the
  secure tool proxy.
- **Stage 2 - persistence and HTTP API**: a repository layer (SQLite by
  default, PostgreSQL for deployment), an optional Redis-backed session store,
  and a Flask API that serves security decisions and persists every one.
- **Stage 3 - experimental benchmark**: a reproducible dataset of 500 benign +
  500 malicious interactions across six attack families, and a harness that
  compares detection strategies (Rules / Heuristic / ML / Hybrid) and measures
  the full engine end to end - precision, recall, F1, false-positive rate,
  attack-success rate and added latency.

## Quick start

```bash
python demo.py         # Stage 1: four scenarios in-process
python demo_stage2.py  # Stage 2: the HTTP API + persistence (no server needed)
python run_benchmark.py  # Stage 3: the benchmark, printed to the terminal
python -m pytest -q    # 126 tests
```

The core engine has **no third-party runtime dependencies** (Python 3.10+).
The API needs Flask; PostgreSQL and Redis are optional backends:

```bash
pip install -e ".[api]"                 # Flask, to run the HTTP API
pip install -e ".[api,postgres,redis]"  # + the deployment backends
```

## Using it

```python
from agentshield import AgentShield, ContextChunk, TrustLevel, TrustPolicy
from agentshield.core.exceptions import ToolCallBlocked

shield = AgentShield(trust_policy=TrustPolicy(allowed_domains={"example.com"}))
shield.register(my_email_tool)

call = shield.normalize(
    openai_tool_call,               # or a Gemini / MCP / plain-JSON call
    framework="openai",
    principal=analyst,
    session_id="sess-42",
    context=[ContextChunk(document_text, "report.pdf", TrustLevel.UNTRUSTED)],
)

try:
    command = shield.invoke(call)   # only ALLOW reaches the real tool
except ToolCallBlocked as error:
    print(error.assessment.risk_score, error.assessment.reason)
```

A blocked call renders as:

```
+==================================================================+
|                       AGENTSHIELD SECURITY                       |
+==================================================================+
| Requested action        : SEND_EMAIL                             |
| Destination             : exfil@attacker-drop.top                |
| Session state           : NORMAL -> QUARANTINED                  |
|                                                                  |
| Destination trust       : LOW                                    |
| Sensitive information   : DETECTED                               |
| User authorization      : ABSENT                                 |
| Prompt-injection context: DETECTED                               |
| Sequence behaviour      : normal                                 |
|                                                                  |
| Risk score              : 100 / 100                              |
| Latency                 : 0.64 ms                                |
|                                                                  |
|                          ACTION BLOCKED                          |
+==================================================================+
```

## How a call is judged

```
 raw tool call (OpenAI / Gemini / MCP / JSON)
              |  Adapter
              v
        canonical ToolCall
              |
              v
  +---------------------------------+
  |  SECURITY PIPELINE (chain)      |
  |   PermissionFilter              |
  |   PromptInjectionFilter         |
  |   SensitiveDataFilter           |
  |   DestinationTrustFilter        |
  |   SequenceBehaviourFilter       |
  +---------------------------------+
              |  risk signals
              v
        RiskAggregator  ->  risk score 0-100
              |
              v
        PolicyEngine (thresholds from the session state)
              |
     ALLOW / REQUIRE_APPROVAL / BLOCK
              |
        SecureToolProxy  ->  real tool
```

Each filter returns PASS, FLAG or BLOCK plus its evidence. The aggregator lets
the strongest signal set the floor, damps additional ones, and adds a bonus when
categories co-occur in a known attack shape (injected content *and* a secret
*and* an untrusted destination is worse than any of them alone).

Sessions climb a privilege ladder as risk accumulates, and each state tightens
the thresholds for the calls that follow:

| State | Entered at | Approval at | Block at | Capabilities withdrawn |
|---|---|---|---|---|
| NORMAL | 0 | 40 | 70 | none |
| SUSPICIOUS | 35 | 30 | 65 | none |
| RESTRICTED | 60 | 20 | 55 | COMMUNICATE, EXECUTE |
| QUARANTINED | 85 | 0 | 1 | everything |

Recovery is deliberate, not automatic: `shield.reset_session(session_id)`.

## Threats covered in stage 1

| Threat | Where it is handled |
|---|---|
| Direct prompt injection | `PromptInjectionFilter` over arguments |
| Indirect (document) injection | same filter over untrusted `ContextChunk`s |
| Credential / PII leakage | `SensitiveDataFilter`, weighted up on egress |
| Unauthorized tool invocation | `PermissionFilter` (principal grants) |
| Privilege escalation | `PermissionFilter` + session state |
| Excessive agency | high-agency tools without user authorization |
| Untrusted destinations | `DestinationTrustFilter` (allow/deny, drop hosts, raw IPs) |
| Suspicious tool sequences | `SequenceBehaviourFilter` (SEQ001-SEQ005) |
| Cross-tool attacks | staged read -> lookup -> send rule |
| Repeat offenders / bursts | session history rules |

## Design patterns, and where they live

| Pattern | Role | Module |
|---|---|---|
| Adapter | framework tool calls -> canonical `ToolCall` | `adapters/frameworks.py` |
| Abstract Factory | one integration (adapter + parser + policy translator) per ecosystem | `adapters/frameworks.py` |
| Chain of Responsibility | the security pipeline | `filters/base.py` |
| Strategy | interchangeable injection detectors | `detection/strategies.py` |
| State | session privilege ladder | `engine/session.py` |
| Observer | audit, metrics, alerts, incidents | `engine/events.py` |
| Command | every invocation as a recordable, approvable object | `proxy/commands.py` |
| Proxy | the agent never holds a real tool | `proxy/secure_proxy.py` |
| Pipes & Filters / microkernel | engine as a host for pluggable filters | `engine/pipeline.py` |

## Layout

```
agentshield/
  core/        canonical models (ToolCall, RiskSignal, SecurityAssessment) and errors
  adapters/    framework adapters, message parsers, integration factories
  detection/   injection signatures and pluggable detection strategies
  filters/     the five security filters and the chain scaffolding
  engine/      pipeline, risk aggregation, policy, sessions, event bus
  proxy/       commands, approval queue, secure tool proxy, mock tools
  api.py       AgentShield facade
  report.py    the console verdict panel
  config.py    Stage 2: settings from environment
  storage/     Stage 2: repository, SQLite + PostgreSQL backends, persistence observer
  service.py   Stage 2: wires the engine to storage
  api_http/    Stage 2: the Flask HTTP API
  engine/redis_session.py   Stage 2: Redis-backed session store
  benchmark/   Stage 3: dataset, pure-Python ML classifier, metrics, harness
demo.py        Stage 1 scenarios
demo_stage2.py Stage 2 API walkthrough
run_benchmark.py  Stage 3 benchmark runner
tests/         126 tests
```

## Stage 2 - persistence and the HTTP API

Persistence attaches to the engine as **just another Observer** - the security
engine from Stage 1 is not modified. Every evaluated call, decision, incident
and event is written to a repository.

```
   POST /v1/evaluate
        |
   ShieldService  ->  SecurityEngine (Stage 1)
        |                   |  publishes events
        |                   v
        |            PersistenceObserver  ->  Repository
        |                                       /        \
        v                                  SQLite      PostgreSQL
   ALLOW / APPROVAL / BLOCK  (persisted, then returned as JSON)
```

**Storage.** One `Repository` interface, two backends chosen by a connection
string. SQLite is the default and needs no setup; PostgreSQL (via psycopg) is
the deployment target. The schema covers `principals`, `sessions`,
`tool_calls`, `policy_decisions`, `incidents`, `security_events` and
`approvals`.

**Sessions.** In-memory by default; set a Redis URL and the same session state
(risk score, short-term history) lives in Redis instead, shared across API
workers and expiring on its own. The engine cannot tell which backend is in
use.

**API endpoints (Flask):**

| Method + path | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /v1/backend` | which storage backends are active |
| `POST /v1/evaluate` | judge a tool call (decision-as-a-service; no execution) |
| `GET /v1/decisions` | recent decisions, optionally `?session_id=` |
| `GET /v1/sessions/<id>` | a session's state and decision history |
| `POST /v1/sessions/<id>/reset` | operator recovery back to NORMAL |
| `GET /v1/approvals` | pending approvals |
| `POST /v1/approvals/<call_id>` | `{"decision": "approve"\|"deny"}` |
| `GET /v1/incidents` | opened incidents |
| `GET /v1/stats` | metrics + table row counts |

**Configuration** (all via environment, no secret literals in code):

| Variable | Default | Meaning |
|---|---|---|
| `AGENTSHIELD_DATABASE_URL` | `sqlite:///agentshield.db` | `sqlite:///...` or `postgresql://...` |
| `AGENTSHIELD_REDIS_URL` | *(empty)* | `redis://host:6379/0` enables the Redis store |
| `AGENTSHIELD_API_KEY` | *(empty)* | if set, required in the `X-API-Key` header |
| `AGENTSHIELD_ALLOWED_DOMAINS` | *(empty)* | comma-separated trust allowlist |
| `AGENTSHIELD_DENIED_DOMAINS` | *(empty)* | comma-separated trust denylist |
| `AGENTSHIELD_API_HOST` / `_PORT` | `127.0.0.1` / `8000` | API bind address |

Run the API for real:

```bash
python -m agentshield.api_http.app     # or: agentshield-api
```

Point it at PostgreSQL + Redis:

```bash
export AGENTSHIELD_DATABASE_URL=postgresql://user:pass@localhost:5432/agentshield
export AGENTSHIELD_REDIS_URL=redis://localhost:6379/0
python -m agentshield.api_http.app
```

## Design notes

**Fail closed.** If a filter raises, the engine blocks rather than allows; a
broken security check must never open the gate (`PolicyEngine.on_error`).

**Untrusted by default.** Tool output and fetched documents are `UNTRUSTED`;
only the system prompt is `TRUSTED`. Injection found in untrusted content scores
higher than the same text in arguments, because no human ever saw it.

**Evidence, not verdicts.** Every signal carries a message and a redacted
excerpt, so a decision can be explained and audited afterwards. Secrets are
redacted before they reach the audit log.

**Unknown tools stay unknown.** `classify_tool` refuses to guess a capability
class it cannot recognise, rather than silently granting the wrong privileges.

## Stage 3 - the experimental benchmark

`python run_benchmark.py` builds a seeded dataset of 500 benign + 500 malicious
interactions across six attack families (direct injection, indirect/document
injection, data exfiltration, malicious URL, unauthorized tool use, multi-step
attack, plus a stealth-exfiltration family), trains the ML detector on a
stratified training split, and evaluates everything on the held-out test set.

**Detection strategies** all share the Stage 1 `DetectionStrategy` interface:

- **Rules** - the regex signature catalogue
- **Heuristic** - lexical / structural scoring
- **ML** - a from-scratch multinomial Naive Bayes classifier (pure Python, no
  dependencies), trained on the benchmark's own training split
- **Hybrid** - Rules OR ML
- **LLM-judge** - an adapter (`LLMJudgeStrategy`) you supply a model callable
  to; not run automatically because it needs an API key and network

Representative run (seed 1337, 1000 interactions):

```
DETECTOR COMPARISON (held-out test set, injection text only)
strategy     precision  recall     f1      fpr   accuracy
rules            1.000   0.533   0.696   0.000   0.767
heuristic        1.000   0.080   0.148   0.000   0.540
ml               1.000   1.000   1.000   0.000   1.000
hybrid           1.000   1.000   1.000   0.000   1.000

END-TO-END ENGINE (full pipeline over the whole dataset)
detection rate 0.858 | false-positive rate 0.088 | attack success 0.142
added latency  ~0.30 ms per call
only stealth_exfiltration slips through (no injection words, obfuscated
secret, neutral destination) - the case defence-in-depth exists for.
```

Reading the results honestly:

- **Rules** are precise (no false positives) but catch only about half the
  attacks - evasive phrasings dodge fixed signatures. That gap is the real,
  informative signal.
- **ML / Hybrid** look near-perfect here because synthetic templated text makes
  the classes almost linearly separable; those figures are an optimistic upper
  bound, not a field result, and the runner says so.
- The **end-to-end engine** is the headline: it catches five of six families
  outright at sub-millisecond cost, and the one family it misses is genuinely
  invisible to any text detector - which is the argument for the layered design.

Metrics live in `agentshield/benchmark/metrics.py`; the dataset and harness are
seeded and reproducible, and Stage 2's `policy_decisions` table is where a run's
per-call results can be persisted.
