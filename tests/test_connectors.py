"""The real tool connectors, against real local servers and real files."""

from __future__ import annotations

import sqlite3

import pytest

from agentshield.config import Settings
from agentshield.proxy.connectors import (
    BlockedAddress,
    HttpRequestTool,
    ListFilesTool,
    ReadFileTool,
    SmtpConfig,
    SmtpEmailTool,
    SqlQueryTool,
    Workspace,
    WriteFileTool,
    build_connectors,
    check_public_host,
)
from tests.servers import SmtpServer, WebServer


@pytest.fixture
def smtp():
    server = SmtpServer()
    yield server
    server.close()


@pytest.fixture
def web():
    server = WebServer()
    yield server
    server.close()


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "app.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, plan TEXT)")
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)",
                     [(i, f"customer {i}", "pro" if i % 2 else "free") for i in range(1, 301)])
    conn.commit()
    conn.close()
    return path


# -- files -------------------------------------------------------------------
def test_files_round_trip_inside_the_workspace(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    write, read, listing = WriteFileTool(workspace), ReadFileTool(workspace), ListFilesTool(workspace)
    assert write.execute(path="notes/today.txt", content="hello") == {
        "written": "notes/today.txt", "bytes": 5}
    assert (tmp_path / "ws" / "notes" / "today.txt").read_text() == "hello"
    assert read.execute(path="notes/today.txt") == "hello"
    assert listing.execute(path="notes") == [
        {"path": "notes/today.txt", "type": "file", "bytes": 5}]


@pytest.mark.parametrize("escape", ["../secret.txt", "/etc/passwd", "a/../../x", "C:/Windows/win.ini"])
def test_no_path_can_leave_the_workspace(tmp_path, escape):
    (tmp_path / "secret.txt").write_text("outside")
    workspace = Workspace(tmp_path / "ws")
    with pytest.raises(PermissionError):
        ReadFileTool(workspace).execute(path=escape)
    with pytest.raises(PermissionError):
        WriteFileTool(workspace).execute(path=escape, content="x")


def test_file_size_is_capped(tmp_path):
    workspace = Workspace(tmp_path, max_bytes=10)
    with pytest.raises(ValueError):
        WriteFileTool(workspace).execute(path="big.txt", content="x" * 11)
    (tmp_path / "big.txt").write_text("x" * 11)
    with pytest.raises(ValueError):
        ReadFileTool(workspace).execute(path="big.txt")
    with pytest.raises(FileNotFoundError):
        ReadFileTool(workspace).execute(path="nope.txt")


# -- email -------------------------------------------------------------------
def test_email_is_delivered_over_smtp(smtp):
    tool = SmtpEmailTool(SmtpConfig(host="127.0.0.1", port=smtp.port,
                                    sender="agent@example.com", starttls=False))
    result = tool.execute(recipient="maya@example.com", cc="sam@example.com",
                          subject="Q3", body="Revenue grew 12%.")
    assert result["sent"] is True
    assert result["recipients"] == ["maya@example.com", "sam@example.com"]
    message = smtp.messages[0]
    assert message["from"] == "agent@example.com"
    assert message["to"] == ["maya@example.com", "sam@example.com"]
    assert "Subject: Q3" in message["data"] and "Revenue grew 12%." in message["data"]


def test_email_rejects_bad_addresses_and_missing_config(smtp):
    tool = SmtpEmailTool(SmtpConfig(host="127.0.0.1", port=smtp.port,
                                    sender="agent@example.com", starttls=False))
    with pytest.raises(ValueError):
        tool.execute(recipient="not-an-address", body="x")
    with pytest.raises(ValueError):
        tool.execute(body="no recipient")
    assert smtp.messages == []
    with pytest.raises(ValueError):
        SmtpEmailTool(SmtpConfig(host="smtp.example.com", sender=""))


# -- database ----------------------------------------------------------------
def test_database_answers_read_only_queries(database):
    tool = SqlQueryTool(f"sqlite:///{database}", max_rows=5)
    result = tool.execute(query="SELECT name FROM customers WHERE plan = ? ORDER BY id",
                          params=["pro"])
    assert result["columns"] == ["name"]
    assert result["rows"][0] == {"name": "customer 1"}
    assert result["row_count"] == 5 and result["truncated"] is True


@pytest.mark.parametrize("statement", [
    "DELETE FROM customers",
    "UPDATE customers SET plan = 'free'",
    "DROP TABLE customers",
])
def test_database_connection_cannot_write(database, statement):
    tool = SqlQueryTool(f"sqlite:///{database}")
    with pytest.raises(sqlite3.OperationalError):
        tool.execute(query=statement)
    assert tool.execute(query="SELECT count(*) AS n FROM customers")["rows"] == [{"n": 300}]


def test_database_allows_one_statement_per_call(database):
    tool = SqlQueryTool(f"sqlite:///{database}")
    with pytest.raises(ValueError):
        tool.execute(query="SELECT 1; DELETE FROM customers")
    with pytest.raises(ValueError):
        tool.execute(query="  ")
    with pytest.raises(ValueError):
        SqlQueryTool("mysql://x")


# -- HTTP --------------------------------------------------------------------
def test_internal_addresses_are_refused_by_default(web):
    tool = HttpRequestTool()
    for url in (web.url + "/json", "http://169.254.169.254/latest/meta-data",
                "http://localhost/"):
        with pytest.raises(BlockedAddress):
            tool.execute(url=url)
    assert web.requests == []
    with pytest.raises(BlockedAddress):
        check_public_host("10.0.0.5")


def test_http_calls_reach_the_server(web):
    tool = HttpRequestTool(allow_private=True)
    got = tool.execute(url=web.url + "/json")
    assert got["status"] == 200 and '"ok": true' in got["body"]
    posted = tool.execute(url=web.url + "/submit", method="POST", body={"a": 1})
    assert posted["status"] == 200
    assert web.requests[-1]["body"] == '{"a": 1}'
    assert web.requests[-1]["headers"]["Content-Type"] == "application/json"
    assert tool.execute(url=web.url + "/missing")["status"] == 404


def test_http_response_is_capped_and_inputs_validated(web):
    tool = HttpRequestTool(allow_private=True, max_bytes=100)
    big = tool.execute(url=web.url + "/big")
    assert len(big["body"]) == 100 and big["truncated"] is True
    with pytest.raises(ValueError):
        tool.execute(url="file:///etc/passwd")
    with pytest.raises(ValueError):
        tool.execute(url=web.url, method="CONNECT")


def test_redirects_are_checked_too(web):
    # the first hop is allowed (a test server on localhost); the redirect to a
    # cloud metadata address must still be refused
    class FirstHopAllowed(HttpRequestTool):
        pass

    tool = FirstHopAllowed(allow_private=False)
    import agentshield.proxy.connectors as connectors

    original = connectors.check_public_host

    def allow_test_server(host, allow_private=False):
        if host == "127.0.0.1":
            return
        original(host, allow_private)

    connectors.check_public_host = allow_test_server
    try:
        with pytest.raises(BlockedAddress):
            tool.execute(url=web.url + "/redirect-internal")
    finally:
        connectors.check_public_host = original


# -- assembly ----------------------------------------------------------------
def test_only_configured_connectors_are_registered(tmp_path):
    settings = Settings(workspace_dir=str(tmp_path), tools="files,email,database,http",
                        smtp_host="", tool_database_url="")
    tools, disabled = build_connectors(settings)
    names = {name for name, _ in tools}
    assert names == {"read_file", "write_file", "list_files", "http_request", "fetch_url"}
    assert "AGENTSHIELD_SMTP_HOST" in disabled["send_email"]
    assert "AGENTSHIELD_TOOL_DATABASE_URL" in disabled["query_database"]


def test_connectors_can_be_switched_off(tmp_path):
    settings = Settings(workspace_dir=str(tmp_path), tools="files")
    tools, disabled = build_connectors(settings)
    assert {name for name, _ in tools} == {"read_file", "write_file", "list_files"}
    assert "[http]" in disabled
