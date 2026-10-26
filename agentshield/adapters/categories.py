"""Tool-name to capability-class mapping.

Every adapter funnels through :func:`classify_tool`, so a tool called
``gmail.send`` in one framework and ``send_email`` in another lands in the same
category and is judged by the same rules.
"""

from __future__ import annotations

import re

from agentshield.core.models import ToolCategory

#: Verbs that send something out.  These always mean COMMUNICATE.
_SEND_VERBS = r"send|post|publish|notify|reply|forward|dm|sms|tweet|share"
#: Channel names.  On their own they suggest COMMUNICATE, but 'GmailReadEmail'
#: reads a mailbox and must not be judged as an outbound send.
_CHANNEL_NOUNS = r"webhook|slack|gmail|mail"
_READ_VERBS = r"read|get|list|search|view|fetch|find|download|check|retrieve"

#: (alternation source, category), most specific class first.
_RULES: tuple[tuple[str, ToolCategory], ...] = (
    (_SEND_VERBS, ToolCategory.COMMUNICATE),
    (_CHANNEL_NOUNS, ToolCategory.COMMUNICATE),
    (r"shell|bash|exec|eval|python|subprocess|terminal|command",
     ToolCategory.EXECUTE),
    (r"write|create|update|delete|remove|move|rename|save|upload|chmod",
     ToolCategory.WRITE_LOCAL),
    (r"sql|query|db|database|table|select|mongo|postgres|redis",
     ToolCategory.QUERY_DATA),
    (r"browse|fetch|http|url|web|search|crawl|scrape|request",
     ToolCategory.READ_REMOTE),
    (r"read|open|list|cat|load|get|find|grep|view|file|dir|folder|contact",
     ToolCategory.READ_LOCAL),
)

#: Compiled with word boundaries, so 'cat' matches 'cat_file' but not
#: 'frobnicate'.
_COMPILED: tuple[tuple[re.Pattern[str], ToolCategory], ...] = tuple(
    (re.compile(rf"\b(?:{alternatives})\w*\b"), category)
    for alternatives, category in _RULES
)

_READ_RE = re.compile(rf"\b(?:{_READ_VERBS})\w*\b")
#: index of the channel-noun rule in ``_RULES``
_CHANNEL_RULE = 1

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(tool_name: str) -> str:
    """'gmail.sendEmail' -> 'gmail send email', so matching is token-wise."""
    spaced = _CAMEL_RE.sub(" ", tool_name)
    return " ".join(part for part in _SPLIT_RE.split(spaced.lower()) if part)


def classify_tool(tool_name: str) -> ToolCategory:
    """Best-effort capability class for a tool name.

    Unknown tools stay :attr:`ToolCategory.UNKNOWN` on purpose - guessing wrong
    would silently hand a tool the wrong privileges.
    """
    name = _tokenize(tool_name)
    reads = bool(_READ_RE.search(name))
    for index, (pattern, category) in enumerate(_COMPILED):
        if index == _CHANNEL_RULE and reads:
            continue
        match = pattern.search(name)
        if match is None:
            continue
        # 'ViewSavedAddresses', 'ListDeletedFiles': after a read verb a past
        # participle describes the data, it is not the action
        if reads and category is ToolCategory.WRITE_LOCAL and match.group(0).endswith("ed"):
            continue
        return category
    return ToolCategory.UNKNOWN


#: Argument names that usually carry the destination of an action, in priority
#: order.
DESTINATION_KEYS: tuple[str, ...] = (
    "recipient", "to", "email", "address", "url", "endpoint", "host",
    "channel", "webhook", "path", "file", "filename", "filepath", "table",
    "database", "target",
)


def infer_destination(arguments: dict) -> str | None:
    """Pull the most likely destination out of a tool call's arguments."""
    lowered = {str(k).lower(): v for k, v in arguments.items()}
    for key in DESTINATION_KEYS:
        for arg_name, value in lowered.items():
            if key == arg_name or arg_name.endswith("_" + key):
                if isinstance(value, (list, tuple)) and value:
                    return str(value[0])
                if value not in (None, ""):
                    return str(value)
    return None
