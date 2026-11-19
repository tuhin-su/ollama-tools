"""
Model Exporter for Ollama.
Supports:
- Portable Archive Bundles (.zip, .tar.gz, .tar)
- Standalone GGUF Model Extraction (.gguf + Modelfile)
- Checksum verification and progress reporting
"""

import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .common import (
    ProgressBar,
    check_directory_access,
    compute_sha256,
    find_model_by_query,
    fix_export_file_ownership,
    format_bytes,
    get_blob_path,
    get_ollama_models_dir,
    get_potential_models_dirs,
    list_models_from_filesystem,
    path_exists_or_restricted,
    sanitize_filename,
)


def extract_metadata_from_layers(models_dir: Path, manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse non-model layers from manifest (template, params, system, license)
    and return them as clean strings/dicts.
    """
    meta: Dict[str, Any] = {
        "template": None,
        "system": None,
        "parameters": {},
        "license": None,
        "model_blob_digest": None,
        "model_blob_size": 0
    }

    layers = manifest_data.get("layers", [])
    largest_size = 0
    largest_digest = None

    for layer in layers:
        media_type = layer.get("mediaType", "")
        digest = layer.get("digest", "")
        size = layer.get("size", 0)

        if size > largest_size:
            largest_size = size
            largest_digest = digest

        blob_path = get_blob_path(models_dir, digest)
        if not blob_path.exists():
            continue

        try:
            raw_bytes = blob_path.read_bytes()
            text = raw_bytes.decode("utf-8", errors="replace")

            if media_type == "application/vnd.ollama.image.model":
                meta["model_blob_digest"] = digest
                meta["model_blob_size"] = size
            elif media_type == "application/vnd.ollama.image.template":
                meta["template"] = text
            elif media_type == "application/vnd.ollama.image.system":
                meta["system"] = text
            elif media_type == "application/vnd.ollama.image.license":
                meta["license"] = text
            elif media_type == "application/vnd.ollama.image.params":
                try:
                    meta["parameters"] = json.loads(text)
                except Exception:
                    meta["parameters"] = text
        except Exception:
            continue

    if not meta["model_blob_digest"] and largest_digest:
        meta["model_blob_digest"] = largest_digest
        meta["model_blob_size"] = largest_size

    return meta


def generate_modelfile_text(meta: Dict[str, Any], model_reference: str = "model.gguf") -> str:
    """Generate Modelfile content from metadata."""
    lines = [f"FROM {model_reference}"]

    if meta.get("template"):
        tmpl = meta["template"].replace('"""', '\\"\\"\\"')
        lines.append(f'TEMPLATE """{tmpl}"""')

    if meta.get("system"):
        sys_p = meta["system"].replace('"""', '\\"\\"\\"')
        lines.append(f'SYSTEM """{sys_p}"""')

    params = meta.get("parameters", {})
    if isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    lines.append(f"PARAMETER {k} {json.dumps(item)}")
            else:
                lines.append(f"PARAMETER {k} {v}")
    elif isinstance(params, str) and params.strip():
        lines.append(params.strip())

    if meta.get("license"):
        lic = meta["license"].replace('"""', '\\"\\"\\"')
        lines.append(f'LICENSE """{lic}"""')

    return "\n".join(lines) + "\n"


def export_model(
    model_identifier: str,
    output_path: Optional[Union[str, Path]] = None,
    output_format: str = "zip",
    gguf_only: bool = False,
    models_dir: Optional[Union[str, Path]] = None,
    compression: bool = False,
    verify_checksums: bool = False,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Export an Ollama model to an archive bundle or standalone GGUF file.

    :param model_identifier: Model name, tag, or ID (e.g. 'llama3:8b', 'bge-m3', '0a109f422b47')
    :param output_path: Destination file or directory path
    :param output_format: 'zip', 'tar', or 'tar.gz' (ignored if gguf_only is True)
    :param gguf_only: If True, only export the GGUF model weights and a Modelfile
    :param models_dir: Custom Ollama models directory
    :param compression: If True, enable compression (ZIP_DEFLATED or gzip)
    :param verify_checksums: Verify SHA256 of all blobs before packaging
    :param quiet: Suppress progress bars and verbose messages
    :return: Export summary dictionary
    """
    resolved_models_dir = get_ollama_models_dir(models_dir)

    # Check directory access
    is_ok, err_msg = check_directory_access(resolved_models_dir)
    if not is_ok:
        raise PermissionError(err_msg)

    # Find the model
    manifest_path, manifest_data, model_name, tag_name = find_model_by_query(resolved_models_dir, model_identifier)

    if not manifest_path or not manifest_data:
        available = list_models_from_filesystem(resolved_models_dir)
        names = [m["name"] for m in available]
        msg = f"No model found matching '{model_identifier}' in {resolved_models_dir}."
        if names:
            msg += f"\nAvailable models: {', '.join(names)}"
        else:
            msg += "\nNo models found in this directory."

        # Check if another candidate directory exists (e.g. Linux service at /usr/share/ollama)
        candidates = get_potential_models_dirs()
        for cand in candidates:
            if cand != resolved_models_dir and path_exists_or_restricted(cand):
                is_accessible, _ = check_directory_access(cand)
                if not is_accessible:
                    msg += (
                        f"\n\n[Diagnostic Note] Ollama system service installation detected at '{cand}' "
                        f"(owned by user 'ollama', uid 997, gid 973).\n"
                        f"If the model you want to export was pulled by the Ollama system service, "
                        f"please run this export command with sudo:\n"
                        f"  sudo python export_model.py {model_identifier}\n"
                        f"Or add your user to the 'ollama' group:\n"
                        f"  sudo usermod -a -G ollama $USER"
                    )
                    break
        raise ValueError(msg)

    config_digest = manifest_data.get("config", {}).get("digest", "")
    clean_digest = config_digest.replace("sha256:", "")
    short_id = clean_digest[:12] if clean_digest else "unknown"

    meta = extract_metadata_from_layers(resolved_models_dir, manifest_data)

    # Determine all blobs to export (deduplicating in case layers share digests)
    layers = manifest_data.get("layers", [])
    raw_digests = [config_digest] + [l["digest"] for l in layers if "digest" in l]
    digests_to_export = list(dict.fromkeys([d for d in raw_digests if d]))

    # Verify blobs exist
    missing_blobs = []
    total_blob_bytes = 0
    for d in digests_to_export:
        bp = get_blob_path(resolved_models_dir, d)
        if not bp.exists():
            missing_blobs.append(bp.name)
        else:
            total_blob_bytes += bp.stat().st_size

    if missing_blobs:
        raise FileNotFoundError(f"Missing required blobs: {', '.join(missing_blobs)}")

    # GGUF-Only Mode
    if gguf_only:
        gguf_digest = meta["model_blob_digest"]
        if not gguf_digest:
            raise ValueError(f"Could not identify GGUF model layer for {model_name}:{tag_name}")

        source_gguf = get_blob_path(resolved_models_dir, gguf_digest)
        safe_name = sanitize_filename(f"{model_name}-{tag_name}")

        if output_path:
            out_p = Path(output_path)
            if out_p.is_dir() or (not out_p.exists() and str(output_path).endswith(('/', '\\'))):
                out_p.mkdir(parents=True, exist_ok=True)
                final_gguf_path = out_p / f"{safe_name}.gguf"
            else:
                final_gguf_path = out_p
        else:
            final_gguf_path = Path.cwd() / f"{safe_name}.gguf"

        if not quiet:
            print(f"Exporting GGUF for {model_name}:{tag_name} ({short_id})...")
            print(f"Source Blob: {source_gguf.name} ({format_bytes(source_gguf.stat().st_size)})")

        file_size = source_gguf.stat().st_size
        pb = ProgressBar(file_size, prefix="Extracting GGUF", quiet=quiet)
        final_gguf_path.parent.mkdir(parents=True, exist_ok=True)

        with open(source_gguf, "rb") as fsrc, open(final_gguf_path, "wb") as fdst:
            while True:
                buf = fsrc.read(4 * 1024 * 1024)
                if not buf:
                    break
                fdst.write(buf)
                pb.update(len(buf))
        pb.finish()

        # Write Modelfile alongside GGUF
        modelfile_path = final_gguf_path.with_suffix(".Modelfile")
        modelfile_content = generate_modelfile_text(meta, model_reference=final_gguf_path.name)
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        # Fix ownership if run under sudo
        fix_export_file_ownership(final_gguf_path)
        fix_export_file_ownership(modelfile_path)

        if not quiet:
            print(f"Exported GGUF to: {final_gguf_path.resolve()}")
            print(f"Exported Modelfile to: {modelfile_path.resolve()}")

        return {
            "mode": "gguf",
            "model": f"{model_name}:{tag_name}",
            "id": short_id,
            "gguf_path": str(final_gguf_path.resolve()),
            "modelfile_path": str(modelfile_path.resolve()),
            "size": file_size
        }

    # Archive Bundle Mode
    safe_name = sanitize_filename(f"{model_name}-{tag_name}-{short_id}")
    fmt = output_format.lower().lstrip(".")
    if fmt not in ("zip", "tar", "tar.gz", "tgz"):
        fmt = "zip"

    ext = f".{fmt}" if not fmt.startswith(".") else fmt
    if fmt == "tgz":
        ext = ".tar.gz"

    if output_path:
        out_p = Path(output_path)
        if out_p.is_dir() or str(output_path).endswith(('/', '\\')):
            out_p.mkdir(parents=True, exist_ok=True)
            final_archive_path = out_p / f"{safe_name}{ext}"
        else:
            final_archive_path = out_p
    else:
        final_archive_path = Path.cwd() / f"{safe_name}{ext}"

    if not quiet:
        print(f"Exporting {model_name}:{tag_name} ({short_id}) to {final_archive_path.name}...")
        print(f"Total model size: {format_bytes(total_blob_bytes)} ({len(digests_to_export)} blobs)")

    # Prepare bundle metadata
    export_meta = {
        "format_version": "1.0",
        "created_by": "ollama-tools",
        "model_name": model_name,
        "tag": tag_name,
        "model_id": short_id,
        "config_digest": config_digest,
        "total_size": total_blob_bytes,
        "blobs": [
            {
                "digest": d,
                "filename": d.replace(":", "-"),
                "size": get_blob_path(resolved_models_dir, d).stat().st_size
            }
            for d in digests_to_export
        ],
        "metadata": {
            "template": meta.get("template"),
            "system": meta.get("system"),
            "parameters": meta.get("parameters"),
            "license": meta.get("license")
        }
    }
    meta_json = json.dumps(export_meta, indent=2).encode("utf-8")
    modelfile_text = generate_modelfile_text(meta, model_reference=meta.get("model_blob_digest", "model.gguf"))
    modelfile_bytes = modelfile_text.encode("utf-8")

    manifest_rel_path = manifest_path.relative_to(resolved_models_dir)
    manifest_rel_posix = manifest_rel_path.as_posix()

    # Create archive
    final_archive_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "zip":
        comp = zipfile.ZIP_DEFLATED if compression else zipfile.ZIP_STORED
        with zipfile.ZipFile(final_archive_path, "w", compression=comp, allowZip64=True) as zf:
            # Write metadata & Modelfile
            zf.writestr("ollama-export.json", meta_json)
            zf.writestr("Modelfile", modelfile_bytes)
            # Write manifest at relative path and root manifest.json (posix forward slashes)
            zf.write(manifest_path, arcname=manifest_rel_posix)
            zf.write(manifest_path, arcname="manifest.json")

            # Stream blobs
            for d in digests_to_export:
                bp = get_blob_path(resolved_models_dir, d)
                blob_size = bp.stat().st_size
                arc_name = f"blobs/{bp.name}"

                if verify_checksums:
                    if not quiet:
                        print(f"Verifying {bp.name}...")
                    expected_hash = d.replace("sha256:", "")
                    actual_hash = compute_sha256(bp)
                    if actual_hash != expected_hash:
                        raise ValueError(f"Checksum mismatch for {bp.name}! Expected {expected_hash}, got {actual_hash}")

                pb = ProgressBar(blob_size, prefix=f"Packing {bp.name[:20]}...", quiet=quiet)
                with zf.open(arc_name, "w", force_zip64=True) as dest_f, open(bp, "rb") as src_f:
                    while True:
                        chunk = src_f.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        dest_f.write(chunk)
                        pb.update(len(chunk))
                pb.finish()

    else:
        # tar or tar.gz
        mode = "w:gz" if (fmt in ("tar.gz", "tgz") or compression) else "w"
        with tarfile.open(final_archive_path, mode) as tf:
            # Helper to add bytes
            def add_bytes(name: str, data: bytes):
                import io
                ti = tarfile.TarInfo(name=name)
                ti.size = len(data)
                ti.mtime = int(manifest_path.stat().st_mtime)
                tf.addfile(ti, io.BytesIO(data))

            add_bytes("ollama-export.json", meta_json)
            add_bytes("Modelfile", modelfile_bytes)
            tf.add(manifest_path, arcname=manifest_rel_posix)
            tf.add(manifest_path, arcname="manifest.json")

            for d in digests_to_export:
                bp = get_blob_path(resolved_models_dir, d)
                blob_size = bp.stat().st_size
                arc_name = f"blobs/{bp.name}"

                ti = tarfile.TarInfo(name=arc_name)
                ti.size = blob_size
                ti.mtime = int(bp.stat().st_mtime)

                pb = ProgressBar(blob_size, prefix=f"Packing {bp.name[:20]}...", quiet=quiet)
                with open(bp, "rb") as src_f:
                    class TarStreamWrapper:
                        def __init__(self, f, pb):
                            self.f = f
                            self.pb = pb
                        def read(self, s=-1):
                            c = self.f.read(s)
                            if c:
                                self.pb.update(len(c))
                            return c
                    tf.addfile(ti, TarStreamWrapper(src_f, pb))
                pb.finish()

    archive_size = final_archive_path.stat().st_size
    fix_export_file_ownership(final_archive_path)

    if not quiet:
        print(f"\nDone! Successfully exported {model_name}:{tag_name}")
        print(f"Archive file: {final_archive_path.resolve()}")
        print(f"Archive size: {format_bytes(archive_size)}")

    return {
        "mode": "bundle",
        "format": fmt,
        "model": f"{model_name}:{tag_name}",
        "id": short_id,
        "archive_path": str(final_archive_path.resolve()),
        "size": archive_size,
        "blobs_count": len(digests_to_export)
    }
