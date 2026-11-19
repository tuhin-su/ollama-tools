#!/usr/bin/env python3
"""
Ollama Model Exporter.

Usage:
    python export_model.py <MODEL_ID_OR_NAME> [options]

Examples:
    python export_model.py 790764642607
    python export_model.py bge-m3:latest -o bge-m3-bundle.zip
    python export_model.py llama3:8b --format tar.gz
    python export_model.py nomic-embed-text --gguf
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_tools.cli import main_export

if __name__ == "__main__":
    sys.exit(main_export())
