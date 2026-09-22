"""AgentShield - a runtime security firewall for tool-using AI agents."""

from agentshield.api import AgentShield
from agentshield.core.exceptions import ApprovalRequired, ToolCallBlocked
from agentshield.core.models import (
    ContextChunk,
    Decision,
    Principal,
    SecurityAssessment,
    ToolCall,
    ToolCategory,
    TrustLevel,
    Verdict,
)
from agentshield.filters.destination_trust import TrustPolicy
from agentshield.report import print_report, render

__version__ = "0.1.0"

__all__ = [
    "AgentShield",
    "ApprovalRequired",
    "ContextChunk",
    "Decision",
    "Principal",
    "SecurityAssessment",
    "ToolCall",
    "ToolCallBlocked",
    "ToolCategory",
    "TrustLevel",
    "TrustPolicy",
    "Verdict",
    "__version__",
    "print_report",
    "render",
]
