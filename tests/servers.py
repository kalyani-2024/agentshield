"""Small real servers the connector tests talk to over sockets.

The connectors under test speak real SMTP and real HTTP; these servers answer
on localhost so the tests need no network and no external account.
"""

from __future__ import annotations

import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class SmtpServer:
    """A minimal SMTP receiver: accepts every message and keeps it."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def reply(self, line: str) -> None:
                self.wfile.write((line + "\r\n").encode())

            def handle(self) -> None:
                self.reply("220 test smtp ready")
                envelope = {"from": None, "to": [], "data": ""}
                while True:
                    line = self.rfile.readline().decode(errors="replace").rstrip("\r\n")
                    if not line:
                        return
                    verb = line.split(" ", 1)[0].upper()
                    if verb in ("EHLO", "HELO"):
                        self.reply("250 test")
                    elif verb == "MAIL":
                        envelope["from"] = line.split(":", 1)[1].strip(" <>")
                        self.reply("250 ok")
                    elif verb == "RCPT":
                        envelope["to"].append(line.split(":", 1)[1].strip(" <>"))
                        self.reply("250 ok")
                    elif verb == "DATA":
                        self.reply("354 end with .")
                        lines = []
                        while True:
                            data = self.rfile.readline().decode(errors="replace")
                            if data in (".\r\n", ".\n", ""):
                                break
                            lines.append(data)
                        envelope["data"] = "".join(lines)
                        outer.messages.append(dict(envelope))
                        envelope = {"from": None, "to": [], "data": ""}
                        self.reply("250 queued")
                    elif verb == "QUIT":
                        self.reply("221 bye")
                        return
                    else:
                        self.reply("250 ok")

        self._server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class WebServer:
    """An HTTP server with a few fixed routes, recording every request."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, body: bytes, ctype: str = "application/json",
                      extra: dict | None = None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for key, value in (extra or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length).decode() if length else ""
                outer.requests.append({"method": self.command, "path": self.path,
                                       "body": body, "headers": dict(self.headers)})
                if self.path == "/json":
                    self._send(200, json.dumps({"ok": True}).encode())
                elif self.path == "/big":
                    self._send(200, b"x" * 5000, "text/plain")
                elif self.path == "/redirect-internal":
                    self._send(302, b"", extra={"Location": "http://169.254.169.254/latest"})
                elif self.path == "/missing":
                    self._send(404, b'{"error": "not found"}')
                else:
                    self._send(200, f"echo {self.command} {self.path}".encode(), "text/plain")

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _handle

            def log_message(self, *args) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
