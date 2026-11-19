"""
Common utilities for Ollama model tools:
- Path discovery (Linux, macOS, Windows)
- Permission handling and ownership fixing (for ollama:x:997:973 system user on Linux)
- Windows / Mac / Linux path sanitization
- Hashing and verification
- Human-readable formatting
- CLI Progress reporting
- Manifest discovery and parsing
"""

import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_OLLAMA_HOST.startswith("http://") and not DEFAULT_OLLAMA_HOST.startswith("https://"):
    DEFAULT_OLLAMA_HOST = f"http://{DEFAULT_OLLAMA_HOST}"


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safely used as a filename across Windows, macOS, and Linux.
    Replaces colons (used in Ollama tags), slashes, and Windows-restricted characters.
    """
    # Replace colons and slashes
    s = name.replace(":", "-").replace("/", "_").replace("\\", "_")
    # Remove or replace invalid Windows characters: < > " | ? *
    s = re.sub(r'[<>\"|?*]', '_', s)
    # Strip leading/trailing whitespace and dots
    s = s.strip(". ")
    return s or "unnamed_model"


def get_potential_models_dirs() -> List[Path]:
    """
    Return list of candidate directories where Ollama stores models
    across Windows, macOS, and Linux.
    """
    candidates = []

    # 1. Explicit environment variable (checked on all OSes)
    env_dir = os.environ.get("OLLAMA_MODELS")
    if env_dir:
        candidates.append(Path(env_dir))

    system = platform.system()

    # 2. Linux specific paths
    if system == "Linux":
        # Official Ollama systemd service path (user: ollama, home: /usr/share/ollama)
        candidates.append(Path("/usr/share/ollama/.ollama/models"))
        candidates.append(Path.home() / ".ollama" / "models")
        candidates.append(Path("/var/lib/ollama/models"))
        candidates.append(Path("/opt/ollama/models"))

    # 3. macOS (Darwin) specific paths
    elif system == "Darwin":
        candidates.append(Path.home() / ".ollama" / "models")
        candidates.append(Path.home() / "Library" / "Application Support" / "Ollama" / "models")
        candidates.append(Path("/Users/Shared/Ollama/models"))

    # 4. Windows specific paths
    elif system == "Windows":
        # Standard user .ollama directory
        candidates.append(Path.home() / ".ollama" / "models")
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            candidates.append(Path(localappdata) / "Ollama" / "models")
            candidates.append(Path(localappdata) / "Programs" / "Ollama" / "models")
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "Ollama" / "models")
        programdata = os.environ.get("PROGRAMDATA")
        if programdata:
            candidates.append(Path(programdata) / "Ollama" / "models")

    # Fallback default for any OS
    candidates.append(Path.home() / ".ollama" / "models")

    # Remove duplicates while preserving priority order
    seen = set()
    result = []
    for c in candidates:
        try:
            resolved = str(c.resolve()) if c.exists() else str(c)
        except Exception:
            resolved = str(c)
        if resolved not in seen:
            seen.add(resolved)
            result.append(c)

    return result


def path_exists_or_restricted(path: Path) -> bool:
    """Check if a path exists, or if any of its parent Ollama directories exist but are restricted."""
    try:
        path.stat()
        return True
    except PermissionError:
        return True
    except OSError:
        pass

    curr = path
    while curr != curr.parent:
        if curr.name in ("ollama", ".ollama"):
            try:
                curr.stat()
                return True
            except PermissionError:
                return True
            except OSError:
                pass
        curr = curr.parent
    return False


def get_ollama_models_dir(custom_path: Optional[Union[str, Path]] = None, require_exists: bool = False) -> Path:
    """
    Resolve the Ollama models directory.
    If custom_path is provided, use it.
    Otherwise, look through standard locations across Linux, macOS, and Windows.
    """
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if require_exists and not p.exists():
            raise FileNotFoundError(f"Specified models directory does not exist: {p}")
        return p

    candidates = get_potential_models_dirs()

    # 1. First check existing and readable directories
    for cand in candidates:
        try:
            if cand.exists():
                list(cand.iterdir())
                return cand
        except (PermissionError, OSError):
            continue

    # 2. Next check if a candidate exists or is restricted by permissions (e.g. /usr/share/ollama)
    for cand in candidates:
        if path_exists_or_restricted(cand):
            return cand

    # Default fallback is ~/.ollama/models
    default_dir = Path.home() / ".ollama" / "models"
    return default_dir


def check_directory_access(models_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Test read/write access to the models directory.
    Returns (is_accessible, error_message).
    """
    if not models_dir.exists():
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            return True, None
        except Exception as e:
            return False, f"Failed to create directory {models_dir}: {e}"

    try:
        # Test read
        list(models_dir.iterdir())
    except PermissionError:
        system = platform.system()
        if system == "Linux" and "/usr/share/ollama" in str(models_dir):
            msg = (
                f"Permission denied accessing '{models_dir}'.\n"
                f"Ollama is running as system service user 'ollama' (uid 997, gid 973).\n"
                f"To access directly on Linux:\n"
                f"  1. Run with sudo: sudo ollama-tools export <model>\n"
                f"  2. Or add your user to the 'ollama' group: sudo usermod -a -G ollama $USER\n"
                f"  3. Or for import, use the API mode without sudo: ollama-tools import <file> --api"
            )
        else:
            msg = f"Permission denied accessing '{models_dir}'. Please check file permissions or run with appropriate privileges."
        return False, msg
    except Exception as e:
        return False, f"Error accessing {models_dir}: {e}"

    return True, None


def fix_ollama_permissions(path: Path) -> None:
    """
    On Linux, if running as root and target is in /usr/share/ollama or system location,
    ensure ownership is set to the 'ollama' user/group so the Ollama service can read it.
    Also ensures appropriate permissions (0o644 for files, 0o755 for dirs).
    """
    if platform.system() != "Linux":
        return

    try:
        import pwd
        # Look up ollama user
        try:
            ollama_user = pwd.getpwnam("ollama")
            uid = ollama_user.pw_uid
            gid = ollama_user.pw_gid
        except KeyError:
            return  # No ollama user on this system

        is_root = (os.geteuid() == 0) if hasattr(os, "geteuid") else False

        if is_root:
            if path.is_file():
                os.chown(path, uid, gid)
                os.chmod(path, 0o644)
            elif path.is_dir():
                os.chown(path, uid, gid)
                os.chmod(path, 0o755)
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        dp = os.path.join(root, d)
                        os.chown(dp, uid, gid)
                        os.chmod(dp, 0o755)
                    for f in files:
                        fp = os.path.join(root, f)
                        os.chown(fp, uid, gid)
                        os.chmod(fp, 0o644)
        else:
            # Regular user: ensure files are readable/writable by group
            if path.is_file():
                os.chmod(path, 0o664)
            elif path.is_dir():
                os.chmod(path, 0o775)
    except Exception:
        pass


def fix_export_file_ownership(filepath: Path) -> None:
    """
    If the exporter is executed with sudo on Linux/macOS, restore ownership of the exported
    archive or GGUF file to the original invoking user (via SUDO_UID / SUDO_GID).
    """
    if platform.system() not in ("Linux", "Darwin"):
        return

    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")

    if sudo_uid and sudo_gid:
        try:
            uid = int(sudo_uid)
            gid = int(sudo_gid)
            os.chown(filepath, uid, gid)
        except Exception:
            pass


def format_bytes(num_bytes: Union[int, float]) -> str:
    """Format bytes to human-readable string (e.g. 1.23 GB)."""
    if num_bytes is None:
        return "Unknown size"
    num = float(num_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0 or unit == 'TB':
            if unit == 'B':
                return f"{int(num)} B"
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds < 0 or seconds > 86400 * 7:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressBar:
    """A clean, dependency-free terminal progress bar."""

    def __init__(self, total_bytes: int, prefix: str = "", unit: str = "bytes", quiet: bool = False, width: int = 30):
        self.total = total_bytes
        self.prefix = prefix
        self.unit = unit
        self.quiet = quiet
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.is_tty = sys.stdout.isatty() and not quiet

    def update(self, advance: int = 0, current: Optional[int] = None):
        """Update progress bar with bytes transferred."""
        if self.quiet:
            return

        if current is not None:
            self.current = current
        else:
            self.current += advance

        now = time.time()
        # Throttle updates to at most once every 0.08s unless finished
        if now - self.last_update_time < 0.08 and self.current < self.total:
            return
        self.last_update_time = now

        elapsed = max(now - self.start_time, 0.001)
        speed = self.current / elapsed
        percent = (self.current / self.total * 100) if self.total > 0 else 0.0
        remaining_sec = ((self.total - self.current) / speed) if (speed > 0 and self.total > self.current) else 0

        filled_len = int(self.width * (self.current / self.total)) if self.total > 0 else 0
        filled_len = min(filled_len, self.width)
        bar = '=' * (filled_len - 1) + '>' if 0 < filled_len < self.width else '=' * filled_len
        bar = bar.ljust(self.width)

        status_str = f" {format_bytes(self.current)} / {format_bytes(self.total)} | {format_bytes(speed)}/s | ETA: {format_time(remaining_sec)}"
        line = f"\r{self.prefix} [{bar}] {percent:5.1f}%{status_str}"

        if self.is_tty:
            sys.stdout.write(line)
            sys.stdout.flush()

    def finish(self, message: Optional[str] = None):
        """Complete the progress display."""
        if self.quiet:
            return
        if self.is_tty:
            now = time.time()
            elapsed = max(now - self.start_time, 0.001)
            speed = self.total / elapsed
            bar = '=' * self.width
            line = f"\r{self.prefix} [{bar}] 100.0% {format_bytes(self.total)} | {format_bytes(speed)}/s | {format_time(elapsed)}"
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        if message:
            print(message)


def compute_sha256(filepath: Union[str, Path], progress_bar: Optional[ProgressBar] = None) -> str:
    """Compute sha256 of a file in 1MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if progress_bar:
                progress_bar.update(len(chunk))
    return h.hexdigest()


def find_manifest_files(models_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Search for all manifest files in models_dir/manifests.
    Returns list of (manifest_path, model_name, tag_name).
    """
    manifests_dir = models_dir / "manifests"
    if not manifests_dir.exists():
        return []

    results = []
    try:
        for file_path in manifests_dir.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(manifests_dir)
                parts = rel.parts
                if len(parts) >= 2:
                    tag_name = parts[-1]
                    model_name = parts[-2]
                    if len(parts) >= 3 and parts[-3] != "library":
                        model_name = f"{parts[-3]}/{model_name}"
                    results.append((file_path, model_name, tag_name))
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied while reading {manifests_dir}. "
            f"Run with sudo or check directory permissions."
        ) from e

    return results


def list_models_from_filesystem(models_dir: Path) -> List[Dict[str, Any]]:
    """
    List all models found in the filesystem models directory with their metadata.
    """
    manifests = find_manifest_files(models_dir)
    models = []

    for manifest_path, model_name, tag in manifests:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_digest = data.get("config", {}).get("digest", "")
            clean_digest = config_digest.replace("sha256:", "")
            model_id = clean_digest[:12] if clean_digest else "unknown"

            # Calculate total size of model and layer digests
            total_size = data.get("config", {}).get("size", 0)
            layers = data.get("layers", [])
            for layer in layers:
                total_size += layer.get("size", 0)

            models.append({
                "name": f"{model_name}:{tag}",
                "model": model_name,
                "tag": tag,
                "id": model_id,
                "digest": clean_digest,
                "size": total_size,
                "manifest_path": manifest_path,
                "manifest_data": data,
                "layers_count": len(layers)
            })
        except Exception:
            continue

    return models


def find_model_by_query(models_dir: Path, query: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Search for a model by:
    1. Exact full name:tag (e.g. "bge-m3:latest" or "llama3:8b")
    2. Model name only (e.g. "bge-m3" -> defaults to latest)
    3. Model ID (e.g. "790764642607" or starting prefix)
    4. Config digest prefix

    Returns (manifest_path, manifest_data, model_name, tag_name).
    """
    models = list_models_from_filesystem(models_dir)
    query_lower = query.lower().strip()

    # 1. Exact match on "model:tag"
    for m in models:
        if m["name"].lower() == query_lower:
            return m["manifest_path"], m["manifest_data"], m["model"], m["tag"]

    # 2. Match without tag (e.g. user entered "bge-m3")
    if ":" not in query_lower:
        # Prefer :latest
        for m in models:
            if m["model"].lower() == query_lower and m["tag"].lower() == "latest":
                return m["manifest_path"], m["manifest_data"], m["model"], m["tag"]
        # Any tag if latest not found
        for m in models:
            if m["model"].lower() == query_lower:
                return m["manifest_path"], m["manifest_data"], m["model"], m["tag"]

    # 3. Match by ID or config digest prefix
    for m in models:
        if m["id"].lower().startswith(query_lower) or m["digest"].lower().startswith(query_lower):
            return m["manifest_path"], m["manifest_data"], m["model"], m["tag"]

    return None, None, None, None


def get_blob_path(models_dir: Path, digest: str) -> Path:
    """Return the filesystem path for a blob digest (sha256:<hash> -> blobs/sha256-<hash>)."""
    clean_name = digest.replace(":", "-")
    return models_dir / "blobs" / clean_name


def read_blob_content(models_dir: Path, digest: str) -> Optional[bytes]:
    """Read contents of a blob if it exists."""
    blob_path = get_blob_path(models_dir, digest)
    if blob_path.exists():
        try:
            return blob_path.read_bytes()
        except Exception:
            return None
    return None
