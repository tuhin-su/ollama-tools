"""
Model Importer for Ollama.
Supports:
- Archive bundles (.zip, .tar, .tar.gz)
- Standalone GGUF files (.gguf)
- Filesystem mode (direct, offline, deduplicated)
- API mode (unprivileged HTTP import via Ollama REST API)
- Model renaming and integrity verification
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .api import OllamaClient
from .common import (
    ProgressBar,
    check_directory_access,
    compute_sha256,
    fix_ollama_permissions,
    format_bytes,
    get_blob_path,
    get_ollama_models_dir,
    sanitize_filename,
)


def inspect_archive(archive_path: Path) -> Dict[str, Any]:
    """
    Inspect an archive bundle without full extraction.
    Returns metadata, manifest data, model name, tag, and list of blob members.
    """
    is_zip = zipfile.is_zipfile(archive_path)
    is_tar = tarfile.is_tarfile(archive_path)

    if not is_zip and not is_tar:
        raise ValueError(f"File is not a valid zip or tar archive: {archive_path}")

    meta = {}
    manifest_data = {}
    blobs = []
    manifest_path_in_arc = None

    if is_zip:
        with zipfile.ZipFile(archive_path, "r") as zf:
            namelist = zf.namelist()

            # Read ollama-export.json if present
            if "ollama-export.json" in namelist:
                try:
                    meta = json.loads(zf.read("ollama-export.json").decode("utf-8"))
                except Exception:
                    pass

            # Read manifest
            if "manifest.json" in namelist:
                try:
                    manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
                    manifest_path_in_arc = "manifest.json"
                except Exception:
                    pass
            else:
                for name in namelist:
                    if "manifests/" in name and not name.endswith("/"):
                        try:
                            manifest_data = json.loads(zf.read(name).decode("utf-8"))
                            manifest_path_in_arc = name
                            break
                        except Exception:
                            continue

            for info in zf.infolist():
                if info.filename.startswith("blobs/") and not info.is_dir():
                    clean_name = Path(info.filename).name
                    digest = clean_name.replace("sha256-", "sha256:")
                    blobs.append({
                        "name": info.filename,
                        "filename": clean_name,
                        "digest": digest,
                        "size": info.file_size
                    })

    else:
        # tar
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name == "ollama-export.json":
                    f = tf.extractfile(member)
                    if f:
                        try:
                            meta = json.loads(f.read().decode("utf-8"))
                        except Exception:
                            pass
                elif member.name == "manifest.json" or ("manifests/" in member.name and member.isfile()):
                    f = tf.extractfile(member)
                    if f and not manifest_data:
                        try:
                            manifest_data = json.loads(f.read().decode("utf-8"))
                            manifest_path_in_arc = member.name
                        except Exception:
                            pass
                elif member.name.startswith("blobs/") and member.isfile():
                    clean_name = Path(member.name).name
                    digest = clean_name.replace("sha256-", "sha256:")
                    blobs.append({
                        "name": member.name,
                        "filename": clean_name,
                        "digest": digest,
                        "size": member.size
                    })

    if not manifest_data and not blobs:
        raise ValueError(f"Archive {archive_path} does not appear to be an Ollama model bundle (no manifest or blobs found).")

    # Determine default model name and tag
    model_name = meta.get("model_name")
    tag_name = meta.get("tag", "latest")

    if not model_name and manifest_path_in_arc:
        parts = Path(manifest_path_in_arc).parts
        if len(parts) >= 2:
            tag_name = parts[-1]
            model_name = parts[-2]
            if len(parts) >= 3 and parts[-3] != "library" and parts[-3] != "manifests":
                model_name = f"{parts[-3]}/{model_name}"

    if not model_name:
        model_name = archive_path.stem.split("-")[0]

    return {
        "is_zip": is_zip,
        "meta": meta,
        "manifest": manifest_data,
        "model_name": model_name,
        "tag": tag_name,
        "blobs": blobs,
        "total_size": sum(b["size"] for b in blobs)
    }


def import_model(
    source_path: Union[str, Path],
    target_name: Optional[str] = None,
    mode: str = "auto",
    models_dir: Optional[Union[str, Path]] = None,
    api_url: Optional[str] = None,
    verify_checksums: bool = True,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Import an Ollama model from an archive bundle (.zip, .tar, .tar.gz) or GGUF file.

    :param source_path: Path to archive bundle or .gguf file
    :param target_name: Optional target model name:tag (e.g. 'custom-llama:v1')
    :param mode: 'auto', 'filesystem', or 'api'
    :param models_dir: Custom Ollama models directory for filesystem mode
    :param api_url: Ollama API URL (e.g. 'http://localhost:11434')
    :param verify_checksums: If True, verify SHA256 checksums
    :param quiet: Suppress progress bars and verbose messages
    :return: Import summary dictionary
    """
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    client = OllamaClient(api_url)

    # Determine mode
    resolved_mode = mode.lower()
    if resolved_mode == "auto":
        # If API is explicitly specified, use API
        if api_url:
            resolved_mode = "api"
        else:
            resolved_models_dir = get_ollama_models_dir(models_dir)
            is_ok, _ = check_directory_access(resolved_models_dir)
            if is_ok:
                resolved_mode = "filesystem"
            elif client.is_online():
                if not quiet:
                    print(f"[Notice] Direct filesystem access to '{resolved_models_dir}' is restricted.")
                    print(f"[Notice] Ollama service is active at {client.host} - automatically switching to API mode (no sudo required)!\n")
                resolved_mode = "api"
            else:
                # Default to filesystem so error message guides the user
                resolved_mode = "filesystem"

    # GGUF Import
    if source.suffix.lower() == ".gguf":
        return _import_gguf(
            gguf_path=source,
            target_name=target_name,
            mode=resolved_mode,
            models_dir=models_dir,
            client=client,
            quiet=quiet
        )

    # Archive Bundle Import
    return _import_bundle(
        archive_path=source,
        target_name=target_name,
        mode=resolved_mode,
        models_dir=models_dir,
        client=client,
        verify_checksums=verify_checksums,
        quiet=quiet
    )


def _import_bundle(
    archive_path: Path,
    target_name: Optional[str],
    mode: str,
    models_dir: Optional[Union[str, Path]],
    client: OllamaClient,
    verify_checksums: bool,
    quiet: bool
) -> Dict[str, Any]:
    """Import an archive bundle (.zip, .tar, .tar.gz)."""
    info = inspect_archive(archive_path)

    # Determine final model name and tag
    if target_name:
        if ":" in target_name:
            final_model, final_tag = target_name.split(":", 1)
        else:
            final_model = target_name
            final_tag = "latest"
    else:
        final_model = info["model_name"]
        final_tag = info["tag"]

    full_target_name = f"{final_model}:{final_tag}"

    if not quiet:
        print(f"Importing model as: {full_target_name}")
        print(f"Bundle size: {format_bytes(info['total_size'])} ({len(info['blobs'])} blobs)")
        print(f"Import mode: {mode.upper()}")

    if mode == "api":
        return _import_bundle_via_api(archive_path, info, full_target_name, client, quiet)
    else:
        return _import_bundle_via_filesystem(archive_path, info, final_model, final_tag, models_dir, verify_checksums, quiet)


def _import_bundle_via_filesystem(
    archive_path: Path,
    info: Dict[str, Any],
    model_name: str,
    tag: str,
    models_dir_param: Optional[Union[str, Path]],
    verify_checksums: bool,
    quiet: bool
) -> Dict[str, Any]:
    """Direct filesystem import into Ollama storage."""
    models_dir = get_ollama_models_dir(models_dir_param)
    is_ok, err_msg = check_directory_access(models_dir)
    if not is_ok:
        raise PermissionError(
            f"{err_msg}\n\nTip: You can import without root permissions via Ollama's API using:\n"
            f"  ollama-tools import '{archive_path}' --api"
        )

    blobs_dir = models_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    # Extract blobs
    extracted_count = 0
    skipped_count = 0
    is_zip = info["is_zip"]

    if is_zip:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for b in info["blobs"]:
                dest_blob_path = blobs_dir / b["filename"]
                if dest_blob_path.exists() and dest_blob_path.stat().st_size == b["size"]:
                    if not quiet:
                        print(f"Blob {b['filename'][:20]}... already exists (deduplicated).")
                    skipped_count += 1
                    continue

                pb = ProgressBar(b["size"], prefix=f"Extracting {b['filename'][:20]}...", quiet=quiet)
                with zf.open(b["name"], "r") as src_f, open(dest_blob_path, "wb") as dst_f:
                    while True:
                        buf = src_f.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        dst_f.write(buf)
                        pb.update(len(buf))
                pb.finish()
                fix_ollama_permissions(dest_blob_path)
                extracted_count += 1

                if verify_checksums:
                    expected = b["digest"].replace("sha256:", "")
                    actual = compute_sha256(dest_blob_path)
                    if actual != expected:
                        dest_blob_path.unlink(missing_ok=True)
                        raise ValueError(f"Checksum mismatch for {b['filename']}! Expected {expected}, got {actual}")
    else:
        # tar
        with tarfile.open(archive_path, "r:*") as tf:
            for b in info["blobs"]:
                dest_blob_path = blobs_dir / b["filename"]
                if dest_blob_path.exists() and dest_blob_path.stat().st_size == b["size"]:
                    if not quiet:
                        print(f"Blob {b['filename'][:20]}... already exists (deduplicated).")
                    skipped_count += 1
                    continue

                member = tf.getmember(b["name"])
                src_f = tf.extractfile(member)
                if not src_f:
                    continue

                pb = ProgressBar(b["size"], prefix=f"Extracting {b['filename'][:20]}...", quiet=quiet)
                with open(dest_blob_path, "wb") as dst_f:
                    while True:
                        buf = src_f.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        dst_f.write(buf)
                        pb.update(len(buf))
                pb.finish()
                fix_ollama_permissions(dest_blob_path)
                extracted_count += 1

                if verify_checksums:
                    expected = b["digest"].replace("sha256:", "")
                    actual = compute_sha256(dest_blob_path)
                    if actual != expected:
                        dest_blob_path.unlink(missing_ok=True)
                        raise ValueError(f"Checksum mismatch for {b['filename']}! Expected {expected}, got {actual}")

    # Write manifest
    # Registry path: manifests/registry.ollama.ai/library/<model>/<tag>
    if "/" in model_name:
        manifest_rel = Path("manifests") / "registry.ollama.ai" / model_name / tag
    else:
        manifest_rel = Path("manifests") / "registry.ollama.ai" / "library" / model_name / tag

    manifest_dest = models_dir / manifest_rel
    manifest_dest.parent.mkdir(parents=True, exist_ok=True)

    manifest_content = json.dumps(info["manifest"], indent=2).encode("utf-8")
    manifest_dest.write_bytes(manifest_content)

    # Ensure ownership and permissions for Linux system service (ollama user)
    fix_ollama_permissions(manifest_dest)
    fix_ollama_permissions(blobs_dir)

    if not quiet:
        print(f"\nManifest written to: {manifest_dest}")
        print(f"Successfully imported {model_name}:{tag} into {models_dir}!")
        print(f"Extracted: {extracted_count} blobs, Deduplicated/Skipped: {skipped_count} blobs.")
        print(f"Run 'ollama list' to view the newly imported model.")

    return {
        "status": "success",
        "mode": "filesystem",
        "model": f"{model_name}:{tag}",
        "manifest_path": str(manifest_dest),
        "blobs_extracted": extracted_count,
        "blobs_skipped": skipped_count
    }


def _import_bundle_via_api(
    archive_path: Path,
    info: Dict[str, Any],
    full_target_name: str,
    client: OllamaClient,
    quiet: bool
) -> Dict[str, Any]:
    """Import an archive bundle via the Ollama HTTP REST API."""
    if not client.is_online():
        raise ConnectionError(
            f"Ollama server is not reachable at {client.host}.\n"
            f"Please ensure Ollama is running ('ollama serve') or specify --models-dir for offline filesystem import."
        )

    # 1. Upload each blob in archive
    is_zip = info["is_zip"]
    blobs = info["blobs"]
    uploaded = 0
    cached = 0

    if is_zip:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for b in blobs:
                digest = b["digest"]
                if client.has_blob(digest):
                    if not quiet:
                        print(f"Blob {b['filename'][:20]}... is already cached on server.")
                    cached += 1
                    continue

                pb = ProgressBar(b["size"], prefix=f"Uploading {b['filename'][:20]}...", quiet=quiet)
                with zf.open(b["name"], "r") as src_f:
                    client.upload_blob(digest, src_f, size=b["size"], progress_bar=pb)
                uploaded += 1
    else:
        with tarfile.open(archive_path, "r:*") as tf:
            for b in blobs:
                digest = b["digest"]
                if client.has_blob(digest):
                    if not quiet:
                        print(f"Blob {b['filename'][:20]}... is already cached on server.")
                    cached += 1
                    continue

                member = tf.getmember(b["name"])
                src_f = tf.extractfile(member)
                if not src_f:
                    continue

                pb = ProgressBar(b["size"], prefix=f"Uploading {b['filename'][:20]}...", quiet=quiet)
                client.upload_blob(digest, src_f, size=b["size"], progress_bar=pb)
                uploaded += 1

    # 2. Identify model weights blob and metadata
    model_digest = None
    meta_dict = info.get("meta", {}).get("metadata", {})

    layers = info.get("manifest", {}).get("layers", [])
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_digest = layer.get("digest")
            break

    if not model_digest:
        # Fallback to largest blob
        if blobs:
            largest = max(blobs, key=lambda x: x["size"])
            model_digest = largest["digest"]

    if not model_digest:
        raise ValueError("Could not determine model weights blob to register in Ollama.")

    if not quiet:
        print(f"\nRegistering model '{full_target_name}' with Ollama API...")

    # Create model in Ollama
    template = meta_dict.get("template")
    system = meta_dict.get("system")
    parameters = meta_dict.get("parameters")
    license_text = meta_dict.get("license")

    res = client.create_model(
        model=full_target_name,
        files={"model.gguf": model_digest},
        template=template,
        system=system,
        parameters=parameters,
        license_text=license_text,
        stream=False
    )

    if not quiet:
        print(f"Successfully imported and registered {full_target_name} via Ollama API!")
        print(f"Blobs Uploaded: {uploaded}, Blobs Cached: {cached}")

    return {
        "status": "success",
        "mode": "api",
        "model": full_target_name,
        "blobs_uploaded": uploaded,
        "blobs_cached": cached,
        "result": res
    }


def _import_gguf(
    gguf_path: Path,
    target_name: Optional[str],
    mode: str,
    models_dir: Optional[Union[str, Path]],
    client: OllamaClient,
    quiet: bool
) -> Dict[str, Any]:
    """Import a standalone GGUF file into Ollama."""
    if not target_name:
        target_name = gguf_path.stem.lower().replace("_", "-") + ":latest"
    elif ":" not in target_name:
        target_name = f"{target_name}:latest"

    file_size = gguf_path.stat().st_size

    if not quiet:
        print(f"Importing GGUF file: {gguf_path.name} ({format_bytes(file_size)})")
        print(f"Target model name: {target_name}")
        print(f"Import mode: {mode.upper()}")

    # Compute digest
    pb = ProgressBar(file_size, prefix="Calculating SHA256", quiet=quiet)
    digest_hex = compute_sha256(gguf_path, progress_bar=pb)
    pb.finish()
    digest = f"sha256:{digest_hex}"

    # Check for accompanying Modelfile if exists
    candidate_modelfiles = [
        gguf_path.with_suffix(".Modelfile"),
        gguf_path.parent / "Modelfile"
    ]
    modelfile_content = None
    for mf in candidate_modelfiles:
        if mf.exists():
            modelfile_content = mf.read_text(encoding="utf-8")
            if not quiet:
                print(f"Found accompanying Modelfile: {mf.name}")
            break

    if mode == "api":
        if not client.is_online():
            raise ConnectionError(f"Ollama server is not online at {client.host}")

        if not client.has_blob(digest):
            pb = ProgressBar(file_size, prefix="Uploading GGUF blob", quiet=quiet)
            client.upload_blob(digest, gguf_path, size=file_size, progress_bar=pb)
        elif not quiet:
            print("Blob already exists on Ollama server.")

        if not quiet:
            print(f"Creating model '{target_name}' in Ollama...")

        client.create_model(
            model=target_name,
            files={"model.gguf": digest},
            modelfile=modelfile_content,
            stream=False
        )

        if not quiet:
            print(f"Successfully imported {target_name} via API!")

        return {
            "status": "success",
            "mode": "api",
            "model": target_name,
            "digest": digest
        }
    else:
        resolved_models_dir = get_ollama_models_dir(models_dir)
        is_ok, err = check_directory_access(resolved_models_dir)
        if not is_ok:
            raise PermissionError(f"{err}\nTip: Run with --api to import without root access.")

        dest_blob = get_blob_path(resolved_models_dir, digest)
        dest_blob.parent.mkdir(parents=True, exist_ok=True)

        if not dest_blob.exists():
            pb = ProgressBar(file_size, prefix="Copying to Ollama store", quiet=quiet)
            with open(gguf_path, "rb") as sf, open(dest_blob, "wb") as df:
                while True:
                    b = sf.read(4 * 1024 * 1024)
                    if not b:
                        break
                    df.write(b)
                    pb.update(len(b))
            pb.finish()
            fix_ollama_permissions(dest_blob)

        # If ollama executable is available, build model using Modelfile
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            mf_temp = tempfile.NamedTemporaryFile("w", delete=False)
            try:
                if modelfile_content:
                    # Update FROM line to point to dest_blob
                    lines = []
                    for line in modelfile_content.splitlines():
                        if line.strip().startswith("FROM "):
                            lines.append(f"FROM {dest_blob}")
                        else:
                            lines.append(line)
                    mf_temp.write("\n".join(lines) + "\n")
                else:
                    mf_temp.write(f"FROM {dest_blob}\n")
                mf_temp.flush()
                mf_temp.close()

                import subprocess
                subprocess.run([ollama_bin, "create", target_name, "-f", mf_temp.name], check=True)
            finally:
                os.unlink(mf_temp.name)

        if not quiet:
            print(f"Successfully imported {target_name}!")

        return {
            "status": "success",
            "mode": "filesystem",
            "model": target_name,
            "blob_path": str(dest_blob)
        }
