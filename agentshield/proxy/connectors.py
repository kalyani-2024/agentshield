"""Real tool connectors: the filesystem, SMTP email, a SQL database, HTTP.

These are what the secure proxy executes once a call is allowed.  Each one acts
on a real resource, so each one also carries the guard rails that belong to
the tool itself rather than to the security pipeline:

==================  ================================================================
read/write/list     confined to one workspace directory; no path may escape it
send_email          real SMTP; recipients are validated, nothing is sent unconfigured
query_database      a read-only connection, one statement per call, a row cap
http_request        http(s) only; private, loopback and link-local addresses refused
                    (including after redirects), response size capped
==================  ================================================================

A connector whose configuration is missing is not registered at all, and
``build_connectors`` reports why, so the API never pretends to have a tool it
cannot run.
"""

from __future__ import annotations

import ipaddress
import json
import re
import smtplib
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import Any

from agentshield.proxy.tools import Tool

_EMAIL_RE = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")


# ======================================================================
# files
# ======================================================================
class Workspace:
    """The one directory file tools may touch."""

    def __init__(self, root: str | Path, max_bytes: int = 1_000_000) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def resolve(self, path: str) -> Path:
        """The real path for ``path``, refusing anything outside the workspace.

        An absolute path, ``..`` segments and symlinks pointing outside all end
        up outside ``root`` once resolved, so one check covers them all.
        """
        if not str(path).strip():
            raise ValueError("a path is required")
        candidate = (self.root / str(path)).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise PermissionError(f"path {path!r} is outside the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file from the workspace"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def execute(self, **arguments: Any) -> Any:
        path = self.workspace.resolve(str(arguments.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(str(arguments.get("path")))
        size = path.stat().st_size
        if size > self.workspace.max_bytes:
            raise ValueError(f"file is {size} bytes; the limit is {self.workspace.max_bytes}")
        return path.read_text(encoding="utf-8", errors="replace")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write a text file in the workspace"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def execute(self, **arguments: Any) -> Any:
        path = self.workspace.resolve(str(arguments.get("path", "")))
        content = str(arguments.get("content", ""))
        data = content.encode("utf-8")
        if len(data) > self.workspace.max_bytes:
            raise ValueError(f"content is {len(data)} bytes; the limit is {self.workspace.max_bytes}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"written": self.workspace.relative(path), "bytes": len(data)}


class ListFilesTool(Tool):
    name = "list_files"
    description = "List the files in a workspace directory"

    def __init__(self, workspace: Workspace, limit: int = 500) -> None:
        self.workspace = workspace
        self.limit = limit

    def execute(self, **arguments: Any) -> Any:
        folder = self.workspace.resolve(str(arguments.get("path") or "."))
        if not folder.is_dir():
            raise NotADirectoryError(str(arguments.get("path")))
        entries = sorted(folder.iterdir(), key=lambda p: p.name)[: self.limit]
        return [
            {
                "path": self.workspace.relative(entry),
                "type": "dir" if entry.is_dir() else "file",
                "bytes": entry.stat().st_size if entry.is_file() else None,
            }
            for entry in entries
        ]


# ======================================================================
# email
# ======================================================================
@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    sender: str = ""
    starttls: bool = True
    use_ssl: bool = False
    timeout: float = 20.0


class SmtpEmailTool(Tool):
    name = "send_email"
    description = "Send an email through the configured SMTP server"

    def __init__(self, config: SmtpConfig) -> None:
        if not config.host:
            raise ValueError("SMTP host is required")
        sender = config.sender or config.username
        if not _EMAIL_RE.match(sender or ""):
            raise ValueError("a valid sender address (AGENTSHIELD_SMTP_FROM) is required")
        self.config = config
        self.sender = sender

    @staticmethod
    def _addresses(value: Any) -> list[str]:
        if value in (None, ""):
            return []
        items = value if isinstance(value, (list, tuple)) else str(value).split(",")
        addresses = [str(item).strip() for item in items if str(item).strip()]
        for address in addresses:
            if not _EMAIL_RE.match(address):
                raise ValueError(f"not an email address: {address!r}")
        return addresses

    def execute(self, **arguments: Any) -> Any:
        to = self._addresses(arguments.get("recipient") or arguments.get("to"))
        cc = self._addresses(arguments.get("cc"))
        if not to:
            raise ValueError("send_email needs a 'recipient'")

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(to)
        if cc:
            message["Cc"] = ", ".join(cc)
        message["Subject"] = str(arguments.get("subject", ""))
        message["Message-ID"] = make_msgid(domain=self.sender.split("@", 1)[1])
        message.set_content(str(arguments.get("body") or arguments.get("message") or ""))

        cfg = self.config
        smtp_class = smtplib.SMTP_SSL if cfg.use_ssl else smtplib.SMTP
        with smtp_class(cfg.host, cfg.port, timeout=cfg.timeout) as server:
            if cfg.starttls and not cfg.use_ssl:
                server.starttls()
            if cfg.username:
                server.login(cfg.username, cfg.password)
            refused = server.send_message(message)
        return {
            "sent": True,
            "recipients": [a for a in to + cc if a not in refused],
            "refused": sorted(refused),
            "message_id": message["Message-ID"],
        }


# ======================================================================
# database
# ======================================================================
def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SqlQueryTool(Tool):
    """Read-only SQL against PostgreSQL or SQLite."""

    name = "query_database"
    description = "Run a read-only SQL query against the configured database"

    def __init__(self, url: str, max_rows: int = 200, timeout_s: int = 10) -> None:
        if not url.startswith(("sqlite:///", "postgresql://", "postgres://")):
            raise ValueError("tool database URL must be sqlite:///... or postgresql://...")
        self.url = url
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    @staticmethod
    def _single_statement(query: str) -> str:
        statement = query.strip().rstrip(";").strip()
        if not statement:
            raise ValueError("query_database needs a 'query'")
        if ";" in statement:
            raise ValueError("one statement per call")
        return statement

    def _connect(self):
        if self.url.startswith("sqlite:///"):
            path = Path(self.url[len("sqlite:///"):]).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"database file {path} does not exist")
            # mode=ro: the connection itself cannot write, whatever the SQL says
            return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True,
                                   timeout=self.timeout_s, check_same_thread=False)
        import psycopg  # lazy: only needed for a PostgreSQL tool database

        conn = psycopg.connect(self.url, connect_timeout=self.timeout_s)
        conn.read_only = True
        conn.execute(f"SET statement_timeout = {int(self.timeout_s * 1000)}")
        return conn

    def execute(self, **arguments: Any) -> Any:
        statement = self._single_statement(str(arguments.get("query") or arguments.get("sql") or ""))
        params = arguments.get("params") or ()
        if isinstance(params, dict):
            params = dict(params)
        elif not isinstance(params, (list, tuple)):
            raise ValueError("'params' must be a list or an object")
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(statement, params)
            columns = [column[0] for column in cursor.description or []]
            fetched = cursor.fetchmany(self.max_rows + 1) if columns else []
        finally:
            conn.close()
        rows = [
            {column: _json_safe(value) for column, value in zip(columns, row)}
            for row in fetched[: self.max_rows]
        ]
        return {"columns": columns, "rows": rows, "row_count": len(rows),
                "truncated": len(fetched) > self.max_rows}


# ======================================================================
# HTTP
# ======================================================================
class BlockedAddress(PermissionError):
    """The URL resolves to an address the firewall must not reach."""


def check_public_host(host: str, allow_private: bool = False) -> None:
    """Refuse hosts that resolve to private, loopback or otherwise internal IPs.

    Without this, an injected "fetch http://169.254.169.254/..." would turn the
    firewall itself into a way into the internal network.
    """
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise ConnectionError(f"cannot resolve {host!r}: {error}") from error
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global or address.is_multicast:
            raise BlockedAddress(f"{host!r} resolves to non-public address {address}")


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def __init__(self, allow_private: bool) -> None:
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme not in ("http", "https"):
            raise BlockedAddress(f"redirect to unsupported scheme {parsed.scheme!r}")
        check_public_host(parsed.hostname or "", self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpRequestTool(Tool):
    name = "http_request"
    description = "Call an external HTTP(S) API or fetch a web page"

    METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")
    #: request headers the caller may not set
    _FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}

    def __init__(self, timeout: float = 15.0, max_bytes: int = 1_000_000,
                 allow_private: bool = False) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.allow_private = allow_private
        self._opener = urllib.request.build_opener(_GuardedRedirects(allow_private))

    def execute(self, **arguments: Any) -> Any:
        url = str(arguments.get("url") or arguments.get("endpoint") or "")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("http_request needs an http(s) 'url'")
        method = str(arguments.get("method", "GET")).upper()
        if method not in self.METHODS:
            raise ValueError(f"unsupported method {method!r}")
        check_public_host(parsed.hostname, self.allow_private)

        headers = {"User-Agent": "agentshield-proxy/1.0"}
        for key, value in (arguments.get("headers") or {}).items():
            if str(key).lower() not in self._FORBIDDEN_HEADERS:
                headers[str(key)] = str(value)
        body = arguments.get("body", arguments.get("data"))
        data = None
        if body is not None and method not in ("GET", "HEAD"):
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            else:
                data = str(body).encode("utf-8")

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            response = self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            response = error  # a 404 or 500 is an answer, not a tool failure
        with response:
            raw = response.read(self.max_bytes + 1)
            charset = response.headers.get_content_charset() or "utf-8"
            return {
                "status": response.status,
                "url": response.geturl(),
                "content_type": response.headers.get("Content-Type", ""),
                "body": raw[: self.max_bytes].decode(charset, errors="replace"),
                "truncated": len(raw) > self.max_bytes,
            }


# ======================================================================
# assembly
# ======================================================================
CONNECTORS = ("files", "email", "database", "http")


def build_connectors(settings) -> tuple[list[tuple[str, Tool]], dict[str, str]]:
    """(registered name, tool) for every enabled connector, plus why others are off."""
    enabled = {name.strip() for name in settings.tools.split(",") if name.strip()}
    tools: list[tuple[str, Tool]] = []
    disabled: dict[str, str] = {}

    if "files" in enabled:
        workspace = Workspace(settings.workspace_dir)
        tools += [("read_file", ReadFileTool(workspace)),
                  ("write_file", WriteFileTool(workspace)),
                  ("list_files", ListFilesTool(workspace))]

    if "email" in enabled:
        if not settings.smtp_host:
            disabled["send_email"] = "set AGENTSHIELD_SMTP_HOST (and credentials) to enable"
        else:
            try:
                tools.append(("send_email", SmtpEmailTool(SmtpConfig(
                    host=settings.smtp_host, port=settings.smtp_port,
                    username=settings.smtp_username, password=settings.smtp_password,
                    sender=settings.smtp_from, starttls=settings.smtp_starttls,
                    use_ssl=settings.smtp_ssl,
                ))))
            except ValueError as error:
                disabled["send_email"] = str(error)

    if "database" in enabled:
        if not settings.tool_database_url:
            disabled["query_database"] = "set AGENTSHIELD_TOOL_DATABASE_URL to enable"
        else:
            try:
                tools.append(("query_database", SqlQueryTool(settings.tool_database_url)))
            except ValueError as error:
                disabled["query_database"] = str(error)

    if "http" in enabled:
        http = HttpRequestTool(timeout=settings.http_timeout,
                               allow_private=settings.http_allow_private)
        tools += [("http_request", http), ("fetch_url", http)]

    for name in set(CONNECTORS) - enabled:
        disabled[f"[{name}]"] = "not listed in AGENTSHIELD_TOOLS"
    return tools, disabled
