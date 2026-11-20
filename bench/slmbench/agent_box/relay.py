"""Runs INSIDE the agent sandbox (system python3, stdlib only): an HTTP endpoint on 127.0.0.1 whose requests travel to the
host over this process's stdin/stdout.

The sandbox has no network at all (deny **); this is the agents' only way to the model, and slmbench on the host decides
what passes (only chat completions, to the model under test). Protocol: one JSON line per message. Request (stdout):
{"id", "method", "path", "body"}; response (stdin): {"id", "status", "ctype", "body"}. Bodies are base64.

usage: python3 -u relay.py PORT
"""
import base64
import http.server
import json
import sys
import threading

PORT = int(sys.argv[1])
_out = threading.Lock()
_waiting: dict = {}            # id -> [Event, response]
_n = [0]
_nlock = threading.Lock()


def _reader():
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        slot = _waiting.get(msg.get("id"))
        if slot:
            slot[1] = msg
            slot[0].set()
    for slot in list(_waiting.values()):     # the host went away: fail every pending request
        slot[1] = {"status": 502, "body": base64.b64encode(b'{"error":"slmbench relay closed"}').decode()}
        slot[0].set()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _relay(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        with _nlock:
            _n[0] += 1
            rid = str(_n[0])
        slot = [threading.Event(), None]
        _waiting[rid] = slot
        with _out:
            sys.stdout.write(json.dumps({"id": rid, "method": self.command, "path": self.path,
                                         "body": base64.b64encode(body).decode()}) + "\n")
            sys.stdout.flush()
        slot[0].wait()
        _waiting.pop(rid, None)
        r = slot[1]
        data = base64.b64decode(r.get("body") or "")
        self.send_response(int(r.get("status") or 502))
        self.send_header("Content-Type", r.get("ctype") or "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _relay


threading.Thread(target=_reader, daemon=True).start()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
sys.stdout.write(json.dumps({"ready": PORT}) + "\n")
sys.stdout.flush()
srv.serve_forever()
