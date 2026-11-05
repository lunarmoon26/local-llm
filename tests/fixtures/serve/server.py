"""Tiny CPU-only fake server for lifecycle tests; never imports an inference engine."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--engine")
p.add_argument("--config")
p.add_argument("--host")
p.add_argument("--port", type=int)
a = p.parse_args()
cfg = json.loads(Path(a.config).read_text())
if cfg.get("test_fail"):
    raise SystemExit(23)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            out = {"status": "ok", "max_context": int(cfg["args"][1])}
        else:
            out = {"data": [{"id": cfg["model_name"]}]}
        self.respond(out)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.respond({"choices": [{"message": {"content": "323"}}]})

    def respond(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTTPServer((a.host, a.port), Handler).serve_forever()
