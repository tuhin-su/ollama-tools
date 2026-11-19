"""
Command Line Interface for ollama-tools.
Subcommands:
  - export: Export an Ollama model to an archive bundle or standalone GGUF
  - import: Import an archive bundle or GGUF into Ollama
  - list: List locally installed Ollama models
  - info: Display detailed metadata for a model or archive bundle
  - verify: Verify integrity (SHA256 checksums) of an archive or installed model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .api import OllamaClient
from .common import (
    compute_sha256,
    format_bytes,
    get_blob_path,
    get_ollama_models_dir,
    list_models_from_filesystem,
)
from .exporter import export_model
from .importer import import_model, inspect_archive


def cmd_export(args: argparse.Namespace) -> int:
    try:
        export_model(
            model_identifier=args.model,
            output_path=args.output,
            output_format=args.format,
            gguf_only=args.gguf,
            models_dir=args.models_dir,
            compression=args.compress,
            verify_checksums=args.verify,
            quiet=args.quiet
        )
        return 0
    except PermissionError as e:
        print(f"\n[Permission Error] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n[Export Error] {e}", file=sys.stderr)
        return 1


def cmd_import(args: argparse.Namespace) -> int:
    try:
        mode = "auto"
        if args.api:
            mode = "api"
        elif args.filesystem:
            mode = "filesystem"

        import_model(
            source_path=args.file,
            target_name=args.name,
            mode=mode,
            models_dir=args.models_dir,
            api_url=args.api_url,
            verify_checksums=not args.no_verify,
            quiet=args.quiet
        )
        return 0
    except PermissionError as e:
        print(f"\n[Permission Error] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n[Import Error] {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    models_dir = get_ollama_models_dir(args.models_dir)

    # If --api flag or models_dir is unreadable, try API
    use_api = args.api
    client = OllamaClient(args.api_url)

    if not use_api:
        try:
            models = list_models_from_filesystem(models_dir)
        except PermissionError:
            if client.is_online():
                use_api = True
            else:
                print(f"[Error] Permission denied accessing {models_dir}.", file=sys.stderr)
                print("Run with sudo or add your user to group ollama, or use --api.", file=sys.stderr)
                return 2

    if use_api:
        if not client.is_online():
            print(f"[Error] Could not connect to Ollama API at {client.host}", file=sys.stderr)
            return 1
        api_models = client.list_models()
        if args.json:
            print(json.dumps(api_models, indent=2))
            return 0

        print(f"{'NAME':<35} {'ID':<15} {'SIZE':<12} {'MODIFIED'}")
        print("-" * 75)
        for m in api_models:
            name = m.get("name", "")
            digest = m.get("digest", "")[:12]
            size = format_bytes(m.get("size", 0))
            modified = m.get("modified_at", "")[:19].replace("T", " ")
            print(f"{name:<35} {digest:<15} {size:<12} {modified}")
        return 0

    if args.json:
        # Strip Path objects for JSON serialization
        json_models = []
        for m in models:
            item = dict(m)
            item["manifest_path"] = str(item["manifest_path"])
            json_models.append(item)
        print(json.dumps(json_models, indent=2))
        return 0

    print(f"Models directory: {models_dir}\n")
    if not models:
        print("No models found.")
        return 0

    print(f"{'NAME':<35} {'ID':<15} {'SIZE':<12} {'LAYERS'}")
    print("-" * 70)
    for m in models:
        name = m.get("name", "")
        model_id = m.get("id", "")
        size = format_bytes(m.get("size", 0))
        layers = str(m.get("layers_count", 0))
        print(f"{name:<35} {model_id:<15} {size:<12} {layers}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve() if Path(args.target).exists() else None

    # Check if target is a file (archive or gguf)
    if target and target.is_file():
        if target.suffix.lower() in (".zip", ".tar", ".gz", ".tgz"):
            try:
                info = inspect_archive(target)
                print(f"Archive: {target.name} ({format_bytes(target.stat().st_size)})")
                print(f"Model Name: {info['model_name']}:{info['tag']}")
                print(f"Total Blobs: {len(info['blobs'])} ({format_bytes(info['total_size'])})")
                meta = info.get("meta", {})
                if meta:
                    print(f"Model ID: {meta.get('model_id', 'N/A')}")
                    print(f"Created By: {meta.get('created_by', 'N/A')}")
                    md = meta.get("metadata", {})
                    if md.get("system"):
                        print(f"System Prompt:\n  {md['system'].strip()}")
                    if md.get("parameters"):
                        print(f"Parameters: {md['parameters']}")
                print("\nBlobs:")
                for b in info["blobs"]:
                    print(f"  {b['filename']} ({format_bytes(b['size'])})")
                return 0
            except Exception as e:
                print(f"[Error inspecting archive] {e}", file=sys.stderr)
                return 1

    # Otherwise query from installed models or API
    models_dir = get_ollama_models_dir(args.models_dir)
    client = OllamaClient(args.api_url)

    if client.is_online():
        try:
            info = client.show_model(args.target)
            details = info.get("details", {})
            print(f"Model: {args.target}")
            print(f"Format: {details.get('format', 'gguf')}")
            print(f"Family: {details.get('family', 'unknown')}")
            print(f"Parameter Size: {details.get('parameter_size', 'unknown')}")
            print(f"Quantization: {details.get('quantization_level', 'unknown')}")
            if info.get("parameters"):
                print(f"\nParameters:\n{info['parameters']}")
            if info.get("template"):
                print(f"\nTemplate:\n{info['template']}")
            return 0
        except Exception:
            pass

    print(f"Could not find model or archive: {args.target}", file=sys.stderr)
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    if not target.exists():
        print(f"File not found: {target}", file=sys.stderr)
        return 1

    if target.suffix.lower() in (".zip", ".tar", ".gz", ".tgz"):
        print(f"Verifying archive bundle: {target.name}...")
        try:
            info = inspect_archive(target)
            is_zip = info["is_zip"]
            errors = []

            if is_zip:
                import zipfile
                with zipfile.ZipFile(target, "r") as zf:
                    for b in info["blobs"]:
                        expected = b["digest"].replace("sha256:", "")
                        print(f"  Checking {b['filename']}... ", end="", flush=True)
                        import hashlib
                        h = hashlib.sha256()
                        with zf.open(b["name"]) as f:
                            while True:
                                chunk = f.read(4 * 1024 * 1024)
                                if not chunk:
                                    break
                                h.update(chunk)
                        actual = h.hexdigest()
                        if actual == expected:
                            print("OK")
                        else:
                            print("FAILED!")
                            errors.append((b["filename"], expected, actual))
            if errors:
                print(f"\nVerification FAILED: {len(errors)} corrupted blobs detected.")
                return 1
            print(f"\nVerification SUCCESSFUL: All {len(info['blobs'])} blobs match their SHA256 checksums.")
            return 0
        except Exception as e:
            print(f"[Verification Error] {e}", file=sys.stderr)
            return 1

    print(f"Unsupported target for verification: {target}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-tools",
        description="Comprehensive toolkit for exporting, importing, and migrating Ollama models."
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Export
    p_export = subparsers.add_parser("export", help="Export an Ollama model to a bundle archive or GGUF")
    p_export.add_argument("model", help="Model name, tag, or ID (e.g. 'llama3:8b', 'bge-m3', '0a109f422b47')")
    p_export.add_argument("-o", "--output", help="Output file path or destination directory")
    p_export.add_argument("-f", "--format", choices=["zip", "tar", "tar.gz"], default="zip", help="Archive format (default: zip)")
    p_export.add_argument("-g", "--gguf", action="store_true", help="Export as raw .gguf file instead of archive bundle")
    p_export.add_argument("-c", "--compress", action="store_true", help="Enable compression (deflate/gzip)")
    p_export.add_argument("-d", "--models-dir", help="Custom Ollama models directory")
    p_export.add_argument("--verify", action="store_true", help="Verify SHA256 checksums before packaging")
    p_export.add_argument("-q", "--quiet", action="store_true", help="Quiet mode (no progress bars)")

    # Import
    p_import = subparsers.add_parser("import", help="Import an archive bundle or GGUF file into Ollama")
    p_import.add_argument("file", help="Path to .zip, .tar, .tar.gz bundle or .gguf model file")
    p_import.add_argument("-n", "--name", help="Optional target model name:tag (e.g. 'my-custom-model:v1')")
    p_import.add_argument("--api", action="store_true", help="Import via Ollama REST API (works without root/sudo)")
    p_import.add_argument("--filesystem", action="store_true", help="Force direct offline filesystem import")
    p_import.add_argument("--api-url", help="Ollama API URL (default: http://localhost:11434)")
    p_import.add_argument("-d", "--models-dir", help="Custom Ollama models directory for filesystem mode")
    p_import.add_argument("--no-verify", action="store_true", help="Skip SHA256 checksum verification")
    p_import.add_argument("-q", "--quiet", action="store_true", help="Quiet mode (no progress bars)")

    # List
    p_list = subparsers.add_parser("list", help="List installed Ollama models")
    p_list.add_argument("-d", "--models-dir", help="Custom Ollama models directory")
    p_list.add_argument("--api", action="store_true", help="List models using Ollama REST API")
    p_list.add_argument("--api-url", help="Ollama API URL (default: http://localhost:11434)")
    p_list.add_argument("--json", action="store_true", help="Output as JSON")

    # Info
    p_info = subparsers.add_parser("info", help="Display details of a model or archive bundle")
    p_info.add_argument("target", help="Model name or path to archive bundle")
    p_info.add_argument("-d", "--models-dir", help="Custom Ollama models directory")
    p_info.add_argument("--api-url", help="Ollama API URL")

    # Verify
    p_verify = subparsers.add_parser("verify", help="Verify integrity of an archive bundle")
    p_verify.add_argument("target", help="Path to archive bundle (.zip, .tar.gz)")

    return parser


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if args.subcommand == "export":
        return cmd_export(args)
    elif args.subcommand == "import":
        return cmd_import(args)
    elif args.subcommand == "list":
        return cmd_list(args)
    elif args.subcommand == "info":
        return cmd_info(args)
    elif args.subcommand == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 0


def main_export() -> int:
    parser = argparse.ArgumentParser(description="Export an Ollama model.")
    parser.add_argument("model", help="Model name, tag, or ID")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-f", "--format", choices=["zip", "tar", "tar.gz"], default="zip")
    parser.add_argument("-g", "--gguf", action="store_true", help="Export as raw GGUF")
    parser.add_argument("-c", "--compress", action="store_true", help="Enable compression")
    parser.add_argument("-d", "--models-dir", help="Models directory")
    parser.add_argument("--verify", action="store_true", help="Verify checksums")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()
    return cmd_export(args)


def main_import() -> int:
    parser = argparse.ArgumentParser(description="Import an Ollama model.")
    parser.add_argument("file", help="Path to archive bundle or GGUF")
    parser.add_argument("-n", "--name", help="Target model name:tag")
    parser.add_argument("--api", action="store_true", help="Import via Ollama REST API")
    parser.add_argument("--filesystem", action="store_true", help="Direct filesystem import")
    parser.add_argument("--api-url", help="Ollama API URL")
    parser.add_argument("-d", "--models-dir", help="Models directory")
    parser.add_argument("--no-verify", action="store_true", help="Skip checksums")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()
    return cmd_import(args)


if __name__ == "__main__":
    sys.exit(main())
