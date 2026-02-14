import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyPDF2 import PdfReader, PdfWriter

import pdf_compressor.engine.split as split_pdf
from pdf_compressor.core.utils import get_file_size_mb


def _make_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as handle:
        writer.write(handle)


def _count_pages(path: Path) -> int:
    with open(path, "rb") as handle:
        reader = PdfReader(handle, strict=False)
        return len(reader.pages)


class TestSplitAccuracy(unittest.TestCase):
    def test_split_by_pages_preserves_page_count(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            input_path = base / "input.pdf"
            _make_pdf(input_path, pages=9)

            parts = split_pdf.split_by_pages(input_path, base, num_parts=3, base_name="input")
            self.assertEqual(len(parts), 3)

            total_pages = sum(_count_pages(p) for p in parts)
            self.assertEqual(total_pages, 9)

    def test_split_by_size_preserves_page_count(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            input_path = base / "input.pdf"
            _make_pdf(input_path, pages=10)

            file_size_mb = get_file_size_mb(input_path)
            if file_size_mb <= 0:
                self.skipTest("Generated PDF size is zero; cannot test split")

            threshold_mb = max(file_size_mb * 0.67, 0.0001)

            def _fake_optimize(src: Path, dest: Path):
                shutil.copy2(src, dest)
                return True, "stub"

            with patch("split_pdf.optimize_split_part", side_effect=_fake_optimize):
                parts = split_pdf.split_by_size(
                    input_path,
                    base,
                    base_name="input",
                    threshold_mb=threshold_mb,
                    skip_optimization_under_threshold=True,
                )

            self.assertGreater(len(parts), 1)
            total_pages = sum(_count_pages(p) for p in parts)
            self.assertEqual(total_pages, 10)
