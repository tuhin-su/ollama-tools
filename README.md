# ollama-tools

A comprehensive, lightweight, and dependency-free toolkit for **exporting**, **importing**, and **migrating** [Ollama](https://ollama.com/) models across machines, environments, and operating systems.

Supports **Linux**, **macOS**, and **Windows** out of the box using only the Python standard library (Python 3.8+).

---

## Highlights

* 📦 **Portable Model Bundles**: Export any Ollama model into a clean, portable bundle archive (`.zip`, `.tar.gz`, `.tar`) containing all manifests, config, layers, and Modelfiles.
* 🎯 **Standalone GGUF Extraction**: Extract raw `.gguf` model weights directly from Ollama along with an automatically reconstructed `.Modelfile` for use with llama.cpp, LM Studio, or other runtimes.
* ⚡ **Deduplicated Import**: Fast offline filesystem import that automatically skips already-existing blobs to save disk space and time.
* 🌐 **API-Based Import (No Sudo Required)**: Import models directly through Ollama's HTTP REST API (`http://localhost:11434`), eliminating permission requirements even on restricted Linux systemd setups (`/usr/share/ollama`).
* 🏷️ **Model Renaming**: Rename models on-the-fly during import (e.g. `--name my-custom-model:v1`).
* 🔒 **Integrity Verification**: Built-in SHA-256 checksum verification during export and import to prevent corrupted model transfers.
* 🖥️ **Cross-Platform**: Fully tested and compatible across **Linux**, **macOS**, and **Windows**. Automatically handles OS-specific storage directories, permissions, and path separators.
* 🚀 **Zero Dependencies**: Pure standard library — no `pip install` required to run the scripts.

---

## Installation

### Option 1: Standalone Scripts (Zero Setup)
Clone the repository and run immediately:
```bash
git clone https://github.com/nic/ollama-tools.git
cd ollama-tools
python3 export_model.py --help
python3 import_model.py --help
```

### Option 2: CLI Package Installation
Install into your Python environment:
```bash
pip install -e .
```
This gives you the global CLI commands:
* `ollama-tools` (unified CLI)
* `ollama-export`
* `ollama-import`

---

## Quick Start

### 1. Exporting a Model

Export by **Model Name and Tag**:
```bash
# Export to a .zip bundle (default)
python3 export_model.py llama3:8b

# Specify custom output path
python3 export_model.py bge-m3:latest -o bge-m3-bundle.zip
```

Export by **Short Model ID** (first 12 characters from `ollama list`):
```bash
python3 export_model.py 790764642607
```

Export as **Raw GGUF** (extracts `.gguf` weights + `.Modelfile`):
```bash
python3 export_model.py nomic-embed-text --gguf
# Creates: nomic-embed-text-latest.gguf and nomic-embed-text-latest.Modelfile
```

Export with **Compression** or as **Tarball**:
```bash
python3 export_model.py llama3:8b --format tar.gz --compress
```

---

### 2. Importing a Model

#### A. Offline Filesystem Import (Default)
Fast, direct extraction into Ollama's local storage:
```bash
python3 import_model.py bge-m3-bundle.zip
```

#### B. Rename During Import
```bash
python3 import_model.py bge-m3-bundle.zip --name my-custom-embeddings:v1
```

#### C. Unprivileged / API Import (No Root / Sudo Required)
When Ollama is running as a system service or on a remote machine:
```bash
python3 import_model.py bge-m3-bundle.zip --api
# Or specify a remote host:
python3 import_model.py bge-m3-bundle.zip --api --api-url http://192.168.1.100:11434
```

#### D. Import a Standalone GGUF File
```bash
python3 import_model.py model.gguf --name custom-model:latest
```

---

### 3. Listing, Inspecting & Verifying

#### List Installed Models:
```bash
ollama-tools list
# Or list via the live Ollama API:
ollama-tools list --api
```

#### Inspect an Exported Bundle:
Inspect model metadata, layers, and system prompt without extracting:
```bash
ollama-tools info bge-m3-bundle.zip
```

#### Verify Bundle Integrity:
Check all blob SHA-256 hashes inside an archive:
```bash
ollama-tools verify bge-m3-bundle.zip
```

---

## Cross-Platform Storage & Permissions Guide

Ollama stores models across manifests and blobs. `ollama-tools` automatically detects the correct directory for your OS:

### 🐧 Linux
| Install Type | Default Storage Path | Permissions / Notes |
| :--- | :--- | :--- |
| **System Service (Official)** | `/usr/share/ollama/.ollama/models` | Runs as user `ollama:ollama` (UID `997`, GID `973`). Directory is mode `750`. |
| **User Mode** | `~/.ollama/models` | Runs under your standard user account. |
| **Custom Path** | Set via `OLLAMA_MODELS` | Configurable in systemd or `.bashrc`. |

#### Working with the Linux `ollama` Service User:
When Ollama is installed via the official `install.sh` script, it runs as the system user `ollama` with home directory `/usr/share/ollama`.

* **Exporting**: If you pull models using the system service, run export with `sudo` or add your user to the `ollama` group:
  ```bash
  sudo python3 export_model.py <MODEL_ID_OR_NAME>
  # Ownership of the exported archive is automatically restored to your user!
  ```
  Or add your user to the group:
  ```bash
  sudo usermod -a -G ollama $USER
  ```
* **Importing**:
  * **Option 1 (Recommended - No Sudo)**: Use `--api` to import via the local Ollama daemon. The daemon writes the files directly with correct permissions:
    ```bash
    python3 import_model.py model-bundle.zip --api
    ```
  * **Option 2 (Filesystem with sudo)**: If you run with `sudo`, `ollama-tools` automatically sets ownership of the imported files to `ollama:ollama` (`997:973`) and permissions to `644`/`755` so the service can immediately access them.

---

### 🍎 macOS
| Install Type | Default Storage Path |
| :--- | :--- |
| **CLI / Homebrew** | `~/.ollama/models` |
| **macOS App** | `~/Library/Application Support/Ollama/models` |

---

### 🪟 Windows
| Install Type | Default Storage Path |
| :--- | :--- |
| **Standard User** | `%USERPROFILE%\.ollama\models` |
| **Local AppData** | `%LOCALAPPDATA%\Ollama\models` |

* Handles Windows filename restrictions (converts `:` in tags to `-`).
* Enforces standard forward slashes (`/`) in ZIP/TAR archives for universal compatibility.

---

## Archive Bundle Structure

An exported bundle contains:
```text
my-model-latest.zip
├── manifest.json                     # Root manifest copy
├── Modelfile                         # Reconstructed Modelfile
├── ollama-export.json                # Human-readable metadata & layer digest mapping
├── manifests/
│   └── registry.ollama.ai/
│       └── library/
│           └── <model>/<tag>         # Standard Ollama manifest hierarchy
└── blobs/
    ├── sha256-<config_hash>          # Model configuration JSON
    ├── sha256-<model_weights_hash>   # GGUF tensor binary
    ├── sha256-<template_hash>        # Prompt template
    └── sha256-<params_hash>          # Model parameters (context, temp, etc.)
```

---

## Running Tests

Run the full cross-platform test suite:
```bash
python3 -m unittest discover tests -v
```

---

## License

MIT License. Free to use and modify for personal and commercial projects.
