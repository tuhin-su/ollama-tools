"""
ollama-tools: A comprehensive toolkit for exporting, importing, and migrating Ollama models.
"""

from .api import OllamaClient
from .common import (
    DEFAULT_OLLAMA_HOST,
    ProgressBar,
    compute_sha256,
    format_bytes,
    get_ollama_models_dir,
    list_models_from_filesystem,
)
from .exporter import export_model
from .importer import import_model, inspect_archive

__version__ = "0.1.0"
__all__ = [
    "export_model",
    "import_model",
    "inspect_archive",
    "OllamaClient",
    "get_ollama_models_dir",
    "list_models_from_filesystem",
    "compute_sha256",
    "format_bytes",
    "ProgressBar",
    "DEFAULT_OLLAMA_HOST",
    "__version__",
]
