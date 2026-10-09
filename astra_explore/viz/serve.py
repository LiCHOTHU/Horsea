"""Local static server for the Astra visualization site: HTTP Range support (video seeking), /health, no caching of HTML.

    python serve.py --root <site dir> --port 8765 --bind 127.0.0.1
Binds to localhost only; reach it from a laptop through an SSH tunnel (see README.md).
"""
import argparse
import json
import os
import re
import sys
import time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            body = json.dumps({"status": "ok", "root": self.directory, "time": time.time()}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if rng and os.path.isfile(path):
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                size = os.path.getsize(path)
                start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
                end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
        super().do_GET()

    def end_headers(self):
        if self.path.endswith((".html", "/")) or self.path == "":
            self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    httpd = ThreadingHTTPServer((a.bind, a.port), partial(Handler, directory=root))
    httpd.daemon_threads = True
    print(f"serving {root} on http://{a.bind}:{a.port}/ (pid {os.getpid()})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
