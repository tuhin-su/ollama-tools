"""
Tests for Ollama API interaction and mock HTTP server import.
"""

import hashlib
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path

from ollama_tools.api import OllamaClient
from ollama_tools.importer import import_model
from ollama_tools.exporter import export_model


class MockOllamaHandler(http.server.BaseHTTPRequestHandler):
    blobs = {}
    created_models = []

    def do_HEAD(self):
        if self.path.startswith("/api/blobs/"):
            digest = self.path.replace("/api/blobs/", "")
            if digest in self.blobs:
                self.send_response(200)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/blobs/"):
            digest = self.path.replace("/api/blobs/", "")
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            self.blobs[digest] = data
            self.send_response(201)
            self.end_headers()
            return

        if self.path == "/api/create":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            self.created_models.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": [{"name": "mock:latest", "size": 100}]}).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logging in tests


class TestApiInteraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MockOllamaHandler.blobs = {}
        MockOllamaHandler.created_models = []
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), MockOllamaHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        cls.api_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_client_endpoints(self):
        client = OllamaClient(self.api_url)
        self.assertTrue(client.is_online())
        models = client.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "mock:latest")

    def test_import_bundle_via_api(self):
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a mock model bundle
            models_dir = Path(temp_dir) / "models"
            blobs_dir = models_dir / "blobs"
            manifests_dir = models_dir / "manifests" / "registry.ollama.ai" / "library" / "api-model"
            blobs_dir.mkdir(parents=True)
            manifests_dir.mkdir(parents=True)

            model_data = b"GGUF_TEST_WEIGHTS"
            m_hash = hashlib.sha256(model_data).hexdigest()
            (blobs_dir / f"sha256-{m_hash}").write_bytes(model_data)

            manifest_data = {
                "schemaVersion": 2,
                "config": {"digest": f"sha256:{m_hash}", "size": len(model_data)},
                "layers": [
                    {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{m_hash}", "size": len(model_data)}
                ]
            }
            (manifests_dir / "latest").write_text(json.dumps(manifest_data), encoding="utf-8")

            bundle_path = Path(temp_dir) / "api-test.zip"
            export_model(
                model_identifier="api-model:latest",
                output_path=bundle_path,
                models_dir=models_dir,
                quiet=True
            )

            # Import using API mode
            client = OllamaClient(self.api_url)
            res = import_model(
                source_path=bundle_path,
                target_name="imported-api-model:v1",
                mode="api",
                api_url=self.api_url,
                quiet=True
            )

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["mode"], "api")
            self.assertEqual(res["model"], "imported-api-model:v1")
            self.assertIn(f"sha256:{m_hash}", MockOllamaHandler.blobs)
            self.assertEqual(len(MockOllamaHandler.created_models), 1)
            created = MockOllamaHandler.created_models[0]
            self.assertEqual(created["model"], "imported-api-model:v1")
            self.assertEqual(created["files"]["model.gguf"], f"sha256:{m_hash}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
