"""Minimal stdlib web form for audit intake (no FastAPI required)."""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from ht_backtest.discovery.pipeline import run_intake_pipeline

FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>HT audit intake</title>
  <style>
    body { font-family: Georgia, serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem;
           background: #f7f3ea; color: #1a1a1a; }
    h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
    .sub { color: #555; margin-bottom: 1.5rem; }
    label { display: block; font-weight: 600; margin-top: 1rem; }
    input, select, textarea { width: 100%; box-sizing: border-box; margin-top: 0.35rem;
      padding: 0.5rem; font: inherit; border: 1px solid #bbb; background: #fff; }
    textarea { min-height: 5rem; }
    button { margin-top: 1.25rem; padding: 0.6rem 1.2rem; font: inherit; cursor: pointer;
      background: #1a1a1a; color: #f7f3ea; border: 0; }
    .note { font-size: 0.9rem; color: #666; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>Strategy audit intake</h1>
  <p class="sub">Plain-English idea → YAML candidate → dry-count → queue or reject.</p>
  <form method="POST" action="/submit">
    <label>Title <input name="title" required placeholder="First killzone raid reclaim"/></label>
    <label>Entry condition (plain English)
      <textarea name="entry_plain" required placeholder="Describe when you enter..."></textarea>
    </label>
    <label>Indicator parameters (JSON object)
      <input name="entry_params" value="{}" placeholder='{"reclaim_bars": 3}'/>
    </label>
    <label>Stop rule
      <textarea name="stop_plain" required placeholder="Stop beyond the sweep extreme"></textarea>
    </label>
    <label>Optional filters (one per line)
      <textarea name="filters" placeholder="Only during London&#10;Volume above average"></textarea>
    </label>
    <label>Timeframe
      <select name="timeframe">
        <option value="15m" selected>15m</option>
        <option value="5m">5m</option>
        <option value="1h">1h</option>
        <option value="4h">4h</option>
        <option value="1d">1d</option>
      </select>
    </label>
    <label>Instrument type
      <select name="instrument_type">
        <option value="crypto" selected>crypto</option>
        <option value="FX">FX</option>
        <option value="stocks">stocks</option>
      </select>
    </label>
    <label>Theoretical spine (why it might beat a coin)
      <textarea name="spine" placeholder="Optional"></textarea>
    </label>
    <label><input type="checkbox" name="fixture" value="1"/> Use offline fixture translator (no Claude API)</label>
    <button type="submit">Submit intake</button>
  </form>
  <p class="note">Claude path needs ANTHROPIC_API_KEY. Crypto 15m only for dry-count MVP.
  Results land under data/candidates/queued or data/candidates/rejected.</p>
</body>
</html>
"""


def _result_html(result: dict) -> str:
    body = html.escape(json.dumps(result, indent=2))
    status = "QUEUED" if result.get("accepted") else "REJECTED"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>{status}</title>
<style>body{{font-family:ui-monospace,monospace;max-width:52rem;margin:2rem auto;padding:0 1rem}}
a{{color:#06c}}</style></head>
<body>
<h1>{html.escape(status)}</h1>
<p>confidence={html.escape(str(result.get('confidence')))} —
<a href="/">new intake</a></p>
<pre>{body}</pre>
</body></html>"""


class IntakeHandler(BaseHTTPRequestHandler):
    fixture_default = False

    def log_message(self, fmt: str, *args) -> None:  # quieter
        print(f"[intake-web] {args[0] if args else fmt}")

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        data = FORM_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/submit":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw, keep_blank_values=True)

        def g(name: str, default: str = "") -> str:
            return (form.get(name) or [default])[0].strip()

        try:
            params = json.loads(g("entry_params") or "{}")
        except json.JSONDecodeError:
            self._html(400, "<h1>Bad indicator parameters JSON</h1>")
            return

        filters = []
        for line in g("filters").splitlines():
            line = line.strip()
            if line:
                filters.append({"plain_english": line, "parameters": {}})

        intake = {
            "title": g("title") or "untitled",
            "entry": {"plain_english": g("entry_plain"), "indicator_parameters": params},
            "stop": {"plain_english": g("stop_plain"), "parameters": {}},
            "filters": filters,
            "timeframe": g("timeframe") or "15m",
            "instrument_type": g("instrument_type") or "crypto",
            "theoretical_spine": g("spine"),
            "source": "web_form",
        }
        fixture = g("fixture") == "1" or self.fixture_default
        try:
            result = run_intake_pipeline(intake, fixture=fixture, max_symbols=None)
        except Exception as exc:  # noqa: BLE001
            self._html(500, f"<h1>Pipeline error</h1><pre>{html.escape(str(exc))}</pre>")
            return
        self._html(200, _result_html(result))

    def _html(self, code: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "127.0.0.1", port: int = 8765, *, fixture_default: bool = False) -> None:
    IntakeHandler.fixture_default = fixture_default
    httpd = ThreadingHTTPServer((host, port), IntakeHandler)
    print(f"Audit intake form: http://{host}:{port}/")
    print("Ctrl+C to stop.")
    httpd.serve_forever()
