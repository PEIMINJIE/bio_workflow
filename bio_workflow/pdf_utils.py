from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pdfplumber


def extract_text_from_pdf(path: Path, max_pages: int = 50) -> str:
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(f"\n\n[PAGE {i + 1}]\n{text}")
    return "\n".join(text_parts)


def extract_ocr_from_pdf(path: Path, max_pages: int = 50, lang: str = "eng") -> str:
    try:
        import fitz  # pymupdf
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return ""

    text_parts = []
    doc = fitz.open(str(path))
    for i in range(min(max_pages, doc.page_count)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            ocr_text = pytesseract.image_to_string(img, lang=lang) or ""
        except Exception:
            ocr_text = ""
        if ocr_text.strip():
            text_parts.append(f"\n\n[PAGE {i + 1} OCR]\n{ocr_text}")
    doc.close()
    return "\n".join(text_parts)


def extract_pdf_content(path: Path, max_pages: int = 50, ocr_enabled: bool = True, lang: str = "eng") -> Tuple[str, str]:
    text = extract_text_from_pdf(path, max_pages=max_pages)
    ocr_text = extract_ocr_from_pdf(path, max_pages=max_pages, lang=lang) if ocr_enabled else ""
    return text, ocr_text


def render_pdf_pages(path: Path, max_pages: int = 50, dpi: int = 200) -> List[bytes]:
    try:
        import fitz  # pymupdf
    except Exception:
        return []

    images: List[bytes] = []
    doc = fitz.open(str(path))
    for i in range(min(max_pages, doc.page_count)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        if getattr(pix, "alpha", 0):
            pix = fitz.Pixmap(pix, 0)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def iter_render_pdf_page_batches(
    path: Path,
    max_pages: int = 50,
    dpi: int = 200,
    batch_size: int = 6,
) -> Iterable[List[bytes]]:
    try:
        import fitz  # pymupdf
    except Exception:
        return []

    if batch_size <= 0:
        batch_size = max_pages

    doc = fitz.open(str(path))
    try:
        total_pages = min(max_pages, doc.page_count)
        batch: List[bytes] = []
        for i in range(total_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            if getattr(pix, "alpha", 0):
                pix = fitz.Pixmap(pix, 0)
            batch.append(pix.tobytes("png"))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        doc.close()


def iter_render_pdf_page_batches_with_plan(
    path: Path,
    batches: List[List[int]],
    dpi: int = 200,
) -> Iterable[Tuple[List[int], List[bytes]]]:
    try:
        import fitz  # pymupdf
    except Exception:
        return []

    doc = fitz.open(str(path))
    try:
        total_pages = doc.page_count
        for batch in batches:
            images: List[bytes] = []
            for i in batch:
                if i < 0 or i >= total_pages:
                    continue
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=dpi)
                if getattr(pix, "alpha", 0):
                    pix = fitz.Pixmap(pix, 0)
                images.append(pix.tobytes("png"))
            if images:
                yield batch, images
    finally:
        doc.close()


def extract_first_page_snippet(path: Path, max_chars: int = 1200, lang: str = "eng") -> str:
    text = ""
    try:
        with pdfplumber.open(str(path)) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
    except Exception:
        text = ""

    text = text.strip()
    if text:
        return text[:max_chars]

    try:
        import fitz  # pymupdf
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return ""

    try:
        doc = fitz.open(str(path))
        try:
            if doc.page_count == 0:
                return ""
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang=lang) or ""
            return ocr_text.strip()[:max_chars]
        finally:
            doc.close()
    except Exception:
        return ""


def detect_keyword_pages(path: Path, max_pages: int, keywords: List[str]) -> List[int]:
    if not keywords:
        return []
    try:
        with pdfplumber.open(str(path)) as pdf:
            total_pages = min(max_pages, len(pdf.pages))
            hits: List[int] = []
            lowered = [k.lower() for k in keywords if k.strip()]
            for i in range(total_pages):
                text = (pdf.pages[i].extract_text() or "").lower()
                if any(k in text for k in lowered):
                    hits.append(i)
            return hits
    except Exception:
        return []


def get_pdf_page_count(path: Path) -> int:
    try:
        import fitz  # pymupdf
    except Exception:
        fitz = None

    if fitz is not None:
        try:
            doc = fitz.open(str(path))
            try:
                return doc.page_count
            finally:
                doc.close()
        except Exception:
            pass

    try:
        with pdfplumber.open(str(path)) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0
