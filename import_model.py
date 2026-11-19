#!/usr/bin/env python3
"""
Ollama Model Importer.

Usage:
    python import_model.py <ARCHIVE_OR_GGUF> [options]

Examples:
    python import_model.py bge-m3-bundle.zip
    python import_model.py model.zip --name custom-model:v1
    python import_model.py model.zip --api
    python import_model.py model.gguf --name my-embed-model
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_tools.cli import main_import

if __name__ == "__main__":
    sys.exit(main_import())
