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

This repository currently contains **stage 1: the core security engine**, which
runs end to end - adapters, the filter pipeline, risk scoring, policy decisions,
session privilege states, the audit trail and the secure tool proxy.

## Quick start

```bash
python demo.py        # four scenarios: benign, indirect injection, multi-step attack, approval
python -m pytest -q   # 85 tests
```

No third-party runtime dependencies; Python 3.10+.

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
demo.py        four end-to-end scenarios
tests/         85 tests
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

## Not in stage 1

Stage 2 adds persistence (PostgreSQL for tool calls, decisions and incidents;
Redis for session risk and rate limits) and an HTTP API. Stage 3 adds the
benchmark: 500 benign and 500 malicious interactions, measuring precision,
recall, F1, false-positive rate, attack success rate and added latency across
the rules / ML / LLM-judge / hybrid strategies. The `DetectionStrategy`
interface and the `MetricsObserver` already exist for exactly that comparison.
