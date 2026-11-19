"""
Ollama HTTP REST API client using pure standard library urllib.
Enables operations without requiring direct filesystem / sudo permissions to /usr/share/ollama.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from .common import DEFAULT_OLLAMA_HOST, ProgressBar, format_bytes


class StreamingReader:
    """A file-like wrapper that reports progress as bytes are read by http.client."""

    def __init__(self, fileobj, total_size: int, progress_bar: Optional[ProgressBar] = None, progress_callback: Optional[Callable[[int], None]] = None):
        self.fileobj = fileobj
        self.total_size = total_size
        self.progress_bar = progress_bar
        self.progress_callback = progress_callback
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.fileobj.read(size)
        if chunk:
            n = len(chunk)
            self.bytes_read += n
            if self.progress_bar:
                self.progress_bar.update(n)
            if self.progress_callback:
                self.progress_callback(n)
        return chunk

    def __len__(self) -> int:
        return self.total_size


class OllamaClient:
    """Client for Ollama HTTP API."""

    def __init__(self, host: Optional[str] = None):
        self.host = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
        if not self.host.startswith("http://") and not self.host.startswith("https://"):
            self.host = f"http://{self.host}"

    def _url(self, endpoint: str) -> str:
        return f"{self.host}/{endpoint.lstrip('/')}"

    def is_online(self) -> bool:
        """Check if Ollama server is running and responding."""
        try:
            req = urllib.request.Request(self._url("/api/tags"), method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models currently installed in Ollama."""
        req = urllib.request.Request(self._url("/api/tags"), method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("models", [])

    def show_model(self, model_name: str) -> Dict[str, Any]:
        """Fetch model metadata, modelfile, parameters, and template from Ollama."""
        body = json.dumps({"name": model_name}).encode("utf-8")
        req = urllib.request.Request(
            self._url("/api/show"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def has_blob(self, digest: str) -> bool:
        """Check if a blob exists on the Ollama server using HEAD /api/blobs/<digest>."""
        if not digest.startswith("sha256:"):
            digest = f"sha256:{digest}"

        url = self._url(f"/api/blobs/{digest}")
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def upload_blob(
        self,
        digest: str,
        source: Union[str, Path, Any],
        size: Optional[int] = None,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> bool:
        """
        Upload a blob to Ollama using POST /api/blobs/<digest>.
        Accepts a filepath, Path, or readable binary stream.
        """
        if not digest.startswith("sha256:"):
            digest = f"sha256:{digest}"

        url = self._url(f"/api/blobs/{digest}")

        # Check if blob is already on the server
        if self.has_blob(digest):
            if progress_bar:
                progress_bar.finish(" (already present on server)")
            return True

        close_file = False
        if isinstance(source, (str, Path)):
            p = Path(source)
            if size is None:
                size = p.stat().st_size
            fileobj = open(p, "rb")
            close_file = True
        else:
            fileobj = source
            if size is None:
                raise ValueError("size must be provided when passing a stream to upload_blob")

        try:
            streamer = StreamingReader(fileobj, size, progress_bar, progress_callback)
            req = urllib.request.Request(
                url,
                data=streamer,
                headers={"Content-Length": str(size)},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                if resp.status in (200, 201):
                    if progress_bar:
                        progress_bar.finish()
                    return True
                return False
        finally:
            if close_file:
                fileobj.close()

    def create_model(
        self,
        model: str,
        files: Optional[Dict[str, str]] = None,
        from_model: Optional[str] = None,
        modelfile: Optional[str] = None,
        template: Optional[str] = None,
        system: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        license_text: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Create a model via POST /api/create.
        """
        payload: Dict[str, Any] = {"model": model, "stream": stream}

        if files:
            payload["files"] = files
        if from_model:
            payload["from"] = from_model
        if modelfile:
            payload["modelfile"] = modelfile
        if template:
            payload["template"] = template
        if system:
            payload["system"] = system
        if parameters:
            payload["parameters"] = parameters
        if license_text:
            payload["license"] = license_text

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url("/api/create"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=600) as resp:
            if stream:
                last_result = {}
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str:
                        data = json.loads(line_str)
                        last_result = data
                        status = data.get("status", "")
                        if "error" in data:
                            raise RuntimeError(f"Ollama create error: {data['error']}")
                return last_result
            else:
                resp_text = resp.read().decode("utf-8")
                return json.loads(resp_text)

    def delete_model(self, model: str) -> bool:
        """Delete a model via DELETE /api/delete."""
        body = json.dumps({"name": model}).encode("utf-8")
        req = urllib.request.Request(
            self._url("/api/delete"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="DELETE"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except urllib.error.HTTPError:
            return False
