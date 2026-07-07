"""Small deployable web/API wrapper for the Orchestrate baseline.

This file intentionally uses only the Python standard library so the demo can
run on Render, Railway, Fly, Docker, or a plain VM without dependency setup.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from main import OUTPUT_COLUMNS, load_user_history, predict_row, read_csv


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
HISTORY = load_user_history(DATASET_DIR / "user_history.csv")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Review Demo</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #1f2933;
    }
    body { margin: 0; }
    main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 28px; }
    h1 { font-size: clamp(28px, 5vw, 48px); line-height: 1; margin: 0 0 12px; letter-spacing: 0; }
    p { line-height: 1.55; color: #52606d; }
    .pill { background: #e0f2fe; color: #075985; padding: 8px 12px; border-radius: 999px; font-size: 13px; white-space: nowrap; }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(340px, 0.9fr); gap: 20px; }
    section, form { background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 18px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05); }
    label { display: block; font-weight: 650; font-size: 13px; margin: 14px 0 6px; color: #334e68; }
    input, textarea, select {
      box-sizing: border-box; width: 100%; border: 1px solid #bcccdc; border-radius: 6px;
      padding: 10px 11px; font: inherit; color: #102a43; background: #fff;
    }
    textarea { min-height: 168px; resize: vertical; }
    button {
      border: 0; border-radius: 6px; padding: 11px 14px; font-weight: 700; cursor: pointer;
      background: #0f766e; color: white; margin-top: 16px;
    }
    button.secondary { background: #334e68; margin-left: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e4e7eb; padding: 9px 8px; text-align: left; vertical-align: top; }
    th { color: #486581; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    code, pre { font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace; }
    pre { overflow: auto; background: #102a43; color: #f0f4f8; border-radius: 8px; padding: 14px; min-height: 220px; }
    .metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0 16px; }
    .metric { background: #f0f4f8; border-radius: 8px; padding: 12px; }
    .metric strong { display: block; font-size: 22px; color: #102a43; }
    @media (max-width: 860px) { header, .grid { display: block; } .pill { display: inline-block; margin-top: 8px; } section { margin-top: 18px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Damage Claim Evidence Review</h1>
        <p>Deterministic demo for reviewing car, laptop, and package damage claims against uploaded image references, user history, and evidence rules.</p>
      </div>
      <span class="pill">No API key required</span>
    </header>
    <div class="grid">
      <form id="claim-form">
        <h2>Try A Claim</h2>
        <label for="user_id">User ID</label>
        <input id="user_id" value="user_002">
        <label for="claim_object">Claim Object</label>
        <select id="claim_object">
          <option>car</option>
          <option>laptop</option>
          <option>package</option>
        </select>
        <label for="image_paths">Image Paths</label>
        <input id="image_paths" value="images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg">
        <label for="user_claim">Claim Conversation</label>
        <textarea id="user_claim">Customer: The front bumper has a scratch and the headlight looks broken. | Support: Should we review both? | Customer: Yes, front bumper and headlight together.</textarea>
        <button type="submit">Predict</button>
        <button class="secondary" type="button" id="run-test">Run Test Set</button>
      </form>
      <section>
        <h2>Result</h2>
        <pre id="result">Submit a claim to see structured evidence review output.</pre>
      </section>
    </div>
    <section style="margin-top:20px">
      <h2>What This Shows</h2>
      <div class="metric-row">
        <div class="metric"><strong>3</strong><span>claim domains</span></div>
        <div class="metric"><strong>14</strong><span>output fields</span></div>
        <div class="metric"><strong>$0</strong><span>model cost</span></div>
      </div>
      <table>
        <tr><th>Decision Area</th><th>What The System Produces</th></tr>
        <tr><td>Evidence</td><td>Whether submitted image references are sufficient and usable.</td></tr>
        <tr><td>Claim Understanding</td><td>Issue type, object part, claim status, and severity.</td></tr>
        <tr><td>Risk Review</td><td>User-history flags, prompt-injection style text, and manual-review triggers.</td></tr>
      </table>
    </section>
  </main>
  <script>
    const result = document.querySelector("#result");
    document.querySelector("#claim-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = {
        user_id: document.querySelector("#user_id").value,
        claim_object: document.querySelector("#claim_object").value,
        image_paths: document.querySelector("#image_paths").value,
        user_claim: document.querySelector("#user_claim").value
      };
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      result.textContent = JSON.stringify(await response.json(), null, 2);
    });
    document.querySelector("#run-test").addEventListener("click", async () => {
      result.textContent = "Running bundled test rows...";
      const response = await fetch("/api/run-test", { method: "POST" });
      result.textContent = JSON.stringify(await response.json(), null, 2);
    });
  </script>
</body>
</html>
"""


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict | list) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            json_response(self, 200, {"status": "ok"})
            return
        if path == "/schema":
            json_response(self, 200, {"columns": OUTPUT_COLUMNS})
            return
        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/predict":
            payload = self.read_json()
            if not isinstance(payload, dict):
                json_response(self, 400, {"error": "Expected a JSON object."})
                return
            row = {
                "user_id": str(payload.get("user_id", "")),
                "image_paths": str(payload.get("image_paths", "")),
                "user_claim": str(payload.get("user_claim", "")),
                "claim_object": str(payload.get("claim_object", "")),
            }
            json_response(self, 200, predict_row(row, HISTORY, DATASET_DIR, strategy="final"))
            return
        if path == "/api/run-test":
            rows = read_csv(DATASET_DIR / "claims.csv")
            predictions = [predict_row(row, HISTORY, DATASET_DIR, strategy="final") for row in rows]
            json_response(self, 200, {"row_count": len(predictions), "preview": predictions[:5]})
            return
        json_response(self, 404, {"error": "not_found"})

    def read_json(self) -> object:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AppHandler)
    print(f"Evidence review demo running on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
