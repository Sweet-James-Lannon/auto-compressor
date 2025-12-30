import ipaddress
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exceptions import DownloadError

import utils


class _FakeResponse:
    def __init__(self, chunks, headers=None, status_code=200, is_redirect=False):
        self._chunks = list(chunks)
        self.headers = headers or {}
        self.status_code = status_code
        self.is_redirect = is_redirect

    def iter_content(self, chunk_size=8192):
        del chunk_size
        for c in self._chunks:
            yield c

    def close(self):
        return None


class TestDownloadPdf(unittest.TestCase):
    def test_download_rejects_when_exceeds_content_length(self):
        response = _FakeResponse(
            chunks=[b"a" * 6, b"b" * 6],
            headers={"content-length": "10"},
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "file.pdf"
            with patch("utils.requests.get", return_value=response), \
                 patch("utils._resolve_hostname_ips", return_value=[ipaddress.ip_address("93.184.216.34")]):
                with self.assertRaises(DownloadError) as ctx:
                    utils.download_pdf("https://example.com/file.pdf", out)
                self.assertEqual(ctx.exception.status_code, 502)
                self.assertFalse(out.exists(), "partial file should be removed on failure")

    def test_download_rejects_when_less_than_content_length(self):
        response = _FakeResponse(
            chunks=[b"a" * 5],
            headers={"content-length": "10"},
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "file.pdf"
            with patch("utils.requests.get", return_value=response), \
                 patch("utils._resolve_hostname_ips", return_value=[ipaddress.ip_address("93.184.216.34")]):
                with self.assertRaises(DownloadError) as ctx:
                    utils.download_pdf("https://example.com/file.pdf", out)
                self.assertEqual(ctx.exception.status_code, 502)
                self.assertFalse(out.exists(), "partial file should be removed on failure")

    def test_download_succeeds_when_matches_content_length(self):
        response = _FakeResponse(
            chunks=[b"a" * 4, b"b" * 6],
            headers={"content-length": "10"},
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "file.pdf"
            with patch("utils.requests.get", return_value=response), \
                 patch("utils._resolve_hostname_ips", return_value=[ipaddress.ip_address("93.184.216.34")]):
                utils.download_pdf("https://example.com/file.pdf", out)
                self.assertTrue(out.exists())
                self.assertEqual(out.stat().st_size, 10)

    def test_download_skips_strict_length_when_content_encoded(self):
        response = _FakeResponse(
            chunks=[b"a" * 12],
            headers={"content-length": "10", "content-encoding": "gzip"},
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "file.pdf"
            with patch("utils.requests.get", return_value=response), \
                 patch("utils._resolve_hostname_ips", return_value=[ipaddress.ip_address("93.184.216.34")]):
                utils.download_pdf("https://example.com/file.pdf", out)
                self.assertTrue(out.exists())
                self.assertEqual(out.stat().st_size, 12)
