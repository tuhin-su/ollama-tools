"""
Comprehensive test suite for ollama-tools.
Tests export, import, inspect, GGUF extraction, deduplication, and CLI parsing.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ollama_tools.common import (
    compute_sha256,
    find_model_by_query,
    format_bytes,
    format_time,
    get_blob_path,
    list_models_from_filesystem,
)
from ollama_tools.exporter import export_model
from ollama_tools.importer import import_model, inspect_archive
from ollama_tools.cli import build_parser


class TestOllamaTools(unittest.TestCase):

    def setUp(self):
        # Create a mock Ollama models directory
        self.temp_dir = tempfile.mkdtemp()
        self.mock_models_dir = Path(self.temp_dir) / "models"
        self.blobs_dir = self.mock_models_dir / "blobs"
        self.manifests_dir = self.mock_models_dir / "manifests" / "registry.ollama.ai" / "library" / "mock-llm"
        self.blobs_dir.mkdir(parents=True)
        self.manifests_dir.mkdir(parents=True)

        # 1. Mock GGUF model blob
        self.model_data = b"GGUF" + b"\x03\x00\x00\x00" + b"mock model weights tensor data 12345"
        self.model_hash = hashlib.sha256(self.model_data).hexdigest()
        (self.blobs_dir / f"sha256-{self.model_hash}").write_bytes(self.model_data)

        # 2. Mock template blob
        self.template_data = b"{{ .System }}\n{{ .Prompt }}"
        self.template_hash = hashlib.sha256(self.template_data).hexdigest()
        (self.blobs_dir / f"sha256-{self.template_hash}").write_bytes(self.template_data)

        # 3. Mock params blob
        self.params_data = json.dumps({"temperature": 0.7, "num_ctx": 4096}).encode("utf-8")
        self.params_hash = hashlib.sha256(self.params_data).hexdigest()
        (self.blobs_dir / f"sha256-{self.params_hash}").write_bytes(self.params_data)

        # 4. Mock config blob
        self.config_data = json.dumps({"architecture": "mock-arch"}).encode("utf-8")
        self.config_hash = hashlib.sha256(self.config_data).hexdigest()
        (self.blobs_dir / f"sha256-{self.config_hash}").write_bytes(self.config_data)

        # 5. Manifest JSON
        self.manifest_data = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": f"sha256:{self.config_hash}",
                "size": len(self.config_data)
            },
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": f"sha256:{self.model_hash}",
                    "size": len(self.model_data)
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": f"sha256:{self.template_hash}",
                    "size": len(self.template_data)
                },
                {
                    "mediaType": "application/vnd.ollama.image.params",
                    "digest": f"sha256:{self.params_hash}",
                    "size": len(self.params_data)
                }
            ]
        }
        (self.manifests_dir / "latest").write_text(json.dumps(self.manifest_data), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_utilities(self):
        self.assertEqual(format_bytes(500), "500 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024 * 5), "5.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 2.5), "2.50 GB")

        self.assertEqual(format_time(45), "00:45")
        self.assertEqual(format_time(125), "02:05")
        self.assertEqual(format_time(3665), "01:01:05")

    def test_model_discovery(self):
        models = list_models_from_filesystem(self.mock_models_dir)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "mock-llm:latest")
        self.assertEqual(models[0]["model"], "mock-llm")
        self.assertEqual(models[0]["tag"], "latest")
        self.assertEqual(models[0]["id"], self.config_hash[:12])

        # Find by name:tag
        path, data, name, tag = find_model_by_query(self.mock_models_dir, "mock-llm:latest")
        self.assertIsNotNone(path)
        self.assertEqual(name, "mock-llm")

        # Find by model name only
        path, data, name, tag = find_model_by_query(self.mock_models_dir, "mock-llm")
        self.assertIsNotNone(path)

        # Find by short ID
        path, data, name, tag = find_model_by_query(self.mock_models_dir, self.config_hash[:8])
        self.assertIsNotNone(path)

        # Non-existent
        path, _, _, _ = find_model_by_query(self.mock_models_dir, "non-existent")
        self.assertIsNone(path)

    def test_export_and_inspect_zip(self):
        out_dir = Path(self.temp_dir) / "output"
        out_dir.mkdir()
        zip_out = out_dir / "test-export.zip"

        result = export_model(
            model_identifier="mock-llm:latest",
            output_path=zip_out,
            output_format="zip",
            models_dir=self.mock_models_dir,
            verify_checksums=True,
            quiet=True
        )

        self.assertTrue(zip_out.exists())
        self.assertEqual(result["model"], "mock-llm:latest")
        self.assertEqual(result["blobs_count"], 4)  # config + 3 layers

        # Inspect archive
        info = inspect_archive(zip_out)
        self.assertTrue(info["is_zip"])
        self.assertEqual(info["model_name"], "mock-llm")
        self.assertEqual(info["tag"], "latest")
        self.assertEqual(len(info["blobs"]), 4)

    def test_export_and_inspect_tar(self):
        out_dir = Path(self.temp_dir) / "output"
        out_dir.mkdir(exist_ok=True)
        tar_out = out_dir / "test-export.tar.gz"

        result = export_model(
            model_identifier="mock-llm",
            output_path=tar_out,
            output_format="tar.gz",
            models_dir=self.mock_models_dir,
            quiet=True
        )

        self.assertTrue(tar_out.exists())
        self.assertEqual(result["format"], "tar.gz")

        info = inspect_archive(tar_out)
        self.assertFalse(info["is_zip"])
        self.assertEqual(info["model_name"], "mock-llm")
        self.assertEqual(len(info["blobs"]), 4)

    def test_export_gguf_only(self):
        out_dir = Path(self.temp_dir) / "output"
        out_dir.mkdir(exist_ok=True)
        gguf_out = out_dir / "custom-model.gguf"

        result = export_model(
            model_identifier="mock-llm",
            output_path=gguf_out,
            gguf_only=True,
            models_dir=self.mock_models_dir,
            quiet=True
        )

        self.assertTrue(gguf_out.exists())
        self.assertEqual(gguf_out.read_bytes(), self.model_data)

        # Check that Modelfile was generated alongside
        modelfile = gguf_out.with_suffix(".Modelfile")
        self.assertTrue(modelfile.exists())
        mf_text = modelfile.read_text(encoding="utf-8")
        self.assertIn("FROM custom-model.gguf", mf_text)
        self.assertIn("TEMPLATE", mf_text)

    def test_import_bundle_filesystem(self):
        # 1. Export first
        zip_path = Path(self.temp_dir) / "model-bundle.zip"
        export_model(
            model_identifier="mock-llm",
            output_path=zip_path,
            models_dir=self.mock_models_dir,
            quiet=True
        )

        # 2. Import into a fresh target models directory
        target_models_dir = Path(self.temp_dir) / "target_models"
        target_models_dir.mkdir()

        res = import_model(
            source_path=zip_path,
            target_name="imported-llm:v2",
            mode="filesystem",
            models_dir=target_models_dir,
            verify_checksums=True,
            quiet=True
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model"], "imported-llm:v2")
        self.assertEqual(res["blobs_extracted"], 4)

        # Verify the model exists in the target directory
        models = list_models_from_filesystem(target_models_dir)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "imported-llm:v2")

        # Test deduplication: importing again should skip all 4 blobs
        res_dedup = import_model(
            source_path=zip_path,
            target_name="imported-llm:v3",
            mode="filesystem",
            models_dir=target_models_dir,
            quiet=True
        )
        self.assertEqual(res_dedup["blobs_extracted"], 0)
        self.assertEqual(res_dedup["blobs_skipped"], 4)

    def test_cli_parser(self):
        parser = build_parser()

        # Export parser
        args = parser.parse_args(["export", "llama3:8b", "-o", "out.zip", "--format", "zip", "-g"])
        self.assertEqual(args.subcommand, "export")
        self.assertEqual(args.model, "llama3:8b")
        self.assertEqual(args.output, "out.zip")
        self.assertTrue(args.gguf)

        # Import parser
        args = parser.parse_args(["import", "bundle.zip", "-n", "renamed:tag", "--api"])
        self.assertEqual(args.subcommand, "import")
        self.assertEqual(args.file, "bundle.zip")
        self.assertEqual(args.name, "renamed:tag")
        self.assertTrue(args.api)

        # List parser
        args = parser.parse_args(["list", "--json"])
        self.assertEqual(args.subcommand, "list")
        self.assertTrue(args.json)


if __name__ == "__main__":
    unittest.main()
