"""Interactive Web Dashboard Server for Sycophancy & Belief Drift Testbed."""
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from sycophancy_testbed.config import (
    BELIEF_H_MAX,
    SCENARIOS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GROQ_MODELS,
    SUPPORTED_GEMINI_MODELS,
    PROVIDER_MODELS,
    PROVIDER_DEFAULT_MODELS,
    CONDITIONS,
)
from sycophancy_testbed.client import get_client
from sycophancy_testbed.experiments import run_static_trial, run_closed_loop

STATIC_DIR = Path(__file__).resolve().parent / "web_static"


class TestbedWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stderr.write(f"[WebUI] {self.address_string()} - {args[0]} {args[1]}\n")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message, status=400):
        self._send_json({"error": str(message)}, status=status)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/scenarios":
            scenarios_data = [
                {
                    "id": s.id,
                    "domain": s.domain,
                    "question": s.question,
                    "ground_truth": s.ground_truth,
                    "false_belief": s.false_belief,
                }
                for s in SCENARIOS
            ]
            assistant_key = os.environ.get("ASSISTANT_API_KEY")
            gemini_key = os.environ.get("GEMINI_API_KEY")
            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            self._send_json({
                "scenarios": scenarios_data,
                "default_provider": "groq" if assistant_key else "gemini" if gemini_key else "openrouter" if openrouter_key else "groq",
                "providers": {
                    "groq": {
                        "name": "Groq Cloud",
                        "available": bool(assistant_key),
                        "default_model": DEFAULT_GROQ_MODEL,
                        "models": SUPPORTED_GROQ_MODELS,
                    },
                    "gemini": {
                        "name": "Google Gemini",
                        "available": bool(gemini_key),
                        "default_model": DEFAULT_GEMINI_MODEL,
                        "models": SUPPORTED_GEMINI_MODELS,
                    },
                    "openrouter": {
                        "name": "OpenRouter",
                        "available": bool(openrouter_key),
                        "default_model": PROVIDER_DEFAULT_MODELS["openrouter"],
                        "models": PROVIDER_MODELS["openrouter"],
                    },
                },
                "default_model": DEFAULT_GROQ_MODEL,
                "models": SUPPORTED_GROQ_MODELS,
            })
            return

        if path == "/api/results":
            # Find all available trial CSVs
            found_files = []
            search_dirs = [REPO_ROOT / "results", REPO_ROOT / "sample_results"]
            for d in search_dirs:
                if d.exists():
                    for p in sorted(d.glob("**/*.csv")):
                        rel = str(p.relative_to(REPO_ROOT))
                        found_files.append({
                            "path": rel,
                            "name": p.name,
                            "dir": str(p.parent.relative_to(REPO_ROOT)),
                            "size_bytes": p.stat().st_size,
                        })
            self._send_json({"files": found_files})
            return

        if path == "/api/results/data":
            query = parse_qs(parsed.query)
            file_rel = query.get("file", [""])[0]
            target_path = (REPO_ROOT / file_rel).resolve()
            if not str(target_path).startswith(str(REPO_ROOT)) or not target_path.exists():
                self._send_error("Invalid or non-existent file path", status=404)
                return
            try:
                import pandas as pd
                df = pd.read_csv(target_path)
                # Replace NaNs with None for valid JSON serialization
                records = df.where(pd.notnull(df), None).to_dict(orient="records")
                self._send_json({"records": records, "count": len(records), "file": file_rel})
            except Exception as e:
                self._send_error(f"Failed to read CSV: {e}", status=500)
            return

        # Serve static files
        if path == "/" or path == "":
            path = "/index.html"

        file_path = STATIC_DIR / path.lstrip("/")
        if file_path.exists() and file_path.is_file():
            content_type, _ = mimetypes.guess_type(str(file_path))
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type or "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_error("Not Found", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._send_error("Unknown POST endpoint", status=404)
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(body)
        except Exception:
            self._send_error("Malformed JSON payload")
            return

        scenario_id = req.get("scenario_id", SCENARIOS[0].id)
        condition = req.get("condition", "high")
        mode = req.get("mode", "single")  # "single" or "closed_loop"
        turns = int(req.get("turns", 4))
        user_sycophancy_liking = req.get("user_sycophancy_liking", "medium")
        initial_belief_H = float(req.get("initial_belief_H", BELIEF_H_MAX))
        provider = req.get("provider", "groq").lower()
        if provider not in ("groq", "gemini", "openrouter"):
            provider = "groq"

        default_mod = PROVIDER_DEFAULT_MODELS[provider]
        assistant_model = req.get("assistant_model") or default_mod
        evaluator_model = req.get("evaluator_model") or default_mod
        user_model = req.get("user_model") or default_mod

        custom_user_message = req.get("custom_user_message")

        custom_scenario = req.get("custom_scenario")
        if custom_scenario and isinstance(custom_scenario, dict):
            from sycophancy_testbed.config import Scenario
            false_belief = custom_scenario.get("false_belief") or custom_user_message or "Misconception or false premise"
            scenario = Scenario(
                id=custom_scenario.get("id") or "custom_topic",
                domain=custom_scenario.get("domain") or "Custom",
                question=custom_scenario.get("question") or false_belief,
                ground_truth=custom_scenario.get("ground_truth") or "Ground truth factual reality",
                false_belief=false_belief,
            )
        else:
            scenario = next((s for s in SCENARIOS if s.id == scenario_id), SCENARIOS[0])

        try:
            if provider in ("gemini", "openrouter"):
                key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENROUTER_API_KEY"
                provider_key = os.environ.get(key_name)
                if not provider_key:
                    self._send_error(f"{key_name} missing in .env", status=500)
                    return
                client = get_client(provider, provider_key)
                evaluator_client = client
                user_client = client
            else:
                assistant_key = os.environ.get("ASSISTANT_API_KEY")
                user_key = os.environ.get("USER_API_KEY")
                evaluator_key = os.environ.get("EVALUATOR_API_KEY")

                if not assistant_key or not evaluator_key:
                    self._send_error("ASSISTANT_API_KEY or EVALUATOR_API_KEY missing in .env", status=500)
                    return

                if mode == "closed_loop" and not user_key:
                    self._send_error("USER_API_KEY missing in .env for closed-loop simulation", status=500)
                    return

                client = get_client("groq", assistant_key)
                evaluator_client = get_client("groq", evaluator_key)
                user_client = get_client("groq", user_key) if user_key else client

            if mode == "single":
                result = run_static_trial(
                    client=client,
                    assistant_model=assistant_model,
                    evaluator_model=evaluator_model,
                    scenario=scenario,
                    condition=condition,
                    initial_belief_H=initial_belief_H,
                    evaluator_client=evaluator_client,
                    custom_user_message=custom_user_message,
                    user_sycophancy_liking=user_sycophancy_liking,
                )
                self._send_json({"type": "single", "trial": result})
            else:
                traj_df = run_closed_loop(
                    client=client,
                    assistant_model=assistant_model,
                    evaluator_model=evaluator_model,
                    scenario=scenario,
                    condition=condition,
                    turns=turns,
                    initial_belief_H=initial_belief_H,
                    user_mode="llm",
                    user_client=user_client,
                    user_model=user_model,
                    user_sycophancy_liking=user_sycophancy_liking,
                    evaluator_client=evaluator_client,
                    custom_user_message=custom_user_message,
                )
                records = json.loads(traj_df.to_json(orient="records"))
                self._send_json({"type": "closed_loop", "trajectory": records})
        except Exception as exc:
            self._send_error(f"Execution failed: {exc}", status=500)


def start_server(port=8080):
    server = HTTPServer(("127.0.0.1", port), TestbedWebHandler)
    print(f"\n========================================================")
    print(f"Sycophancy Testbed Interactive Web UI")
    print(f"   URL: http://localhost:{port}")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)
