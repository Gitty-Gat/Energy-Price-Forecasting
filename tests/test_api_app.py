from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestApiApp(unittest.TestCase):
    def test_uvicorn_boots_and_serves_health_and_forecast_routes(self) -> None:
        port = _find_free_port()
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            health_payload = None
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                        health_payload = json.loads(resp.read().decode("utf-8"))
                        self.assertEqual(resp.status, 200)
                        break
                except Exception:
                    time.sleep(0.2)

            if health_payload is None:
                output = ""
                if proc.stdout is not None:
                    try:
                        output = proc.stdout.read()
                    except Exception:
                        output = ""
                self.fail(f"uvicorn did not become healthy in time. Output:\n{output}")

            self.assertEqual(health_payload["status"], "ok")
            self.assertIn("timestamp_utc", health_payload)

            request = Request(
                f"http://127.0.0.1:{port}/forecast/run",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=1) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(resp.status, 200)
                self.assertEqual(payload["status"], "accepted")
                self.assertIn("detail", payload)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            if proc.stdout is not None:
                proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
