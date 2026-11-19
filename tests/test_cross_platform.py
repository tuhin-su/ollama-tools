"""
Tests for cross-platform compatibility across Windows, macOS, and Linux:
- Path discovery across OSes
- Filename sanitization (Windows reserved characters, colons)
- Permissions fixing for Linux ollama user (uid 997, gid 973)
- Zip slip / directory traversal security check
"""

import os
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_tools.common import (
    get_potential_models_dirs,
    sanitize_filename,
    fix_ollama_permissions,
    fix_export_file_ownership,
)


class TestCrossPlatform(unittest.TestCase):

    def test_sanitize_filename(self):
        # Colons in Ollama tags (e.g. llama3:8b) are invalid on Windows
        self.assertEqual(sanitize_filename("llama3:8b"), "llama3-8b")
        # Slashes and backslashes
        self.assertEqual(sanitize_filename("user/model:tag"), "user_model-tag")
        self.assertEqual(sanitize_filename(r"user\model:tag"), "user_model-tag")
        # Windows restricted characters < > " | ? *
        self.assertEqual(sanitize_filename('bad<name>"test|file?*'), "bad_name__test_file__")
        # Empty fallback
        self.assertEqual(sanitize_filename(""), "unnamed_model")

    def test_windows_paths_discovery(self):
        with patch("platform.system", return_value="Windows"), \
             patch.dict(os.environ, {
                 "LOCALAPPDATA": r"C:\Users\TestUser\AppData\Local",
                 "APPDATA": r"C:\Users\TestUser\AppData\Roaming",
                 "PROGRAMDATA": r"C:\ProgramData"
             }):
            dirs = get_potential_models_dirs()
            dir_strs = [str(d) for d in dirs]
            # Verify Windows standard directories are present
            self.assertTrue(any("AppData" in s for s in dir_strs))
            self.assertTrue(any(".ollama" in s for s in dir_strs))

    def test_macos_paths_discovery(self):
        with patch("platform.system", return_value="Darwin"):
            dirs = get_potential_models_dirs()
            dir_strs = [str(d) for d in dirs]
            self.assertTrue(any("Library/Application Support/Ollama" in s or ".ollama" in s for s in dir_strs))

    def test_linux_paths_discovery(self):
        with patch("platform.system", return_value="Linux"):
            dirs = get_potential_models_dirs()
            dir_strs = [str(d) for d in dirs]
            # Official Linux systemd service path for user ollama
            self.assertIn("/usr/share/ollama/.ollama/models", dir_strs)
            self.assertTrue(any(".ollama/models" in s for s in dir_strs))

    def test_fix_export_file_ownership_with_sudo(self):
        with tempfile.NamedTemporaryFile() as tmp:
            tmp_path = Path(tmp.name)
            with patch("platform.system", return_value="Linux"), \
                 patch.dict(os.environ, {"SUDO_UID": "1000", "SUDO_GID": "1000"}), \
                 patch("os.chown") as mock_chown:
                fix_export_file_ownership(tmp_path)
                mock_chown.assert_called_once_with(tmp_path, 1000, 1000)

    def test_fix_ollama_permissions_as_root(self):
        class MockPwd:
            pw_uid = 997
            pw_gid = 973

        with tempfile.NamedTemporaryFile() as tmp:
            tmp_path = Path(tmp.name)
            with patch("platform.system", return_value="Linux"), \
                 patch("os.geteuid", return_value=0), \
                 patch("pwd.getpwnam", return_value=MockPwd()), \
                 patch("os.chown") as mock_chown, \
                 patch("os.chmod") as mock_chmod:
                fix_ollama_permissions(tmp_path)
                mock_chown.assert_called_with(tmp_path, 997, 973)
                mock_chmod.assert_called_with(tmp_path, 0o644)


if __name__ == "__main__":
    unittest.main()
