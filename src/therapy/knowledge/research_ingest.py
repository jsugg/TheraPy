"""Bounded, fully local extraction and OCR for owner-curated research sources."""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image

MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_SOURCE_PAGES = 300
MAX_SOURCE_CHARS = 5_000_000
MAX_IMAGE_PIXELS = 40_000_000
OCR_REVIEW_THRESHOLD = 0.75

_SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}
_IGNORED_HTML = {"script", "style", "nav", "header", "footer", "form", "svg"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class OCRBlock:
    """One locally recognized text region."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """Stable citation block extracted from a source."""

    anchor: str
    text: str
    extraction_method: str
    page: int | None = None
    heading: str | None = None
    confidence: float | None = None
    needs_review: bool = False
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class ExtractedSource:
    """Normalized source content plus extraction audit metadata."""

    format: str
    media_type: str
    markdown: str
    blocks: tuple[SourceBlock, ...]
    status: str
    ocr_metadata: dict[str, object]


class OCRBackend(Protocol):
    """Local OCR boundary used by image and scanned-PDF extractors."""

    @property
    def metadata(self) -> dict[str, object]: ...

    def recognize(self, image: Image) -> list[OCRBlock]: ...


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.casefold()).strip("-")
    return slug[:80] or "section"


def _decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise ValueError("text source contains NUL bytes")
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("text source must be UTF-8 encoded")
    if len(text) > MAX_SOURCE_CHARS:
        raise ValueError("source exceeds character limit")
    return text


def _normalize(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def _validate_input(data: bytes, filename: str, declared_type: str | None) -> str:
    if not data:
        raise ValueError("source file is empty")
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds 25 MiB limit")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must not contain a path")
    suffix = Path(filename).suffix.casefold()
    format_ = _SUPPORTED_EXTENSIONS.get(suffix)
    if format_ is None:
        raise ValueError("unsupported source format")
    if format_ == "pdf" and not data.startswith(b"%PDF-"):
        raise ValueError("PDF extension does not match file signature")
    if format_ == "image":
        valid_signature = (
            data.startswith(b"\x89PNG\r\n\x1a\n")
            if suffix == ".png"
            else data.startswith(b"\xff\xd8\xff")
            if suffix in {".jpg", ".jpeg"}
            else data.startswith(b"RIFF") and data[8:12] == b"WEBP"
            if suffix == ".webp"
            else data.startswith((b"II*\x00", b"MM\x00*"))
        )
        if not valid_signature:
            raise ValueError("image extension does not match file signature")
    declared = (declared_type or "application/octet-stream").split(";", 1)[0].strip()
    if declared not in {"", "application/octet-stream"}:
        compatible = (
            declared == "application/pdf"
            if format_ == "pdf"
            else declared
            in {
                ".png": {"image/png"},
                ".jpg": {"image/jpeg"},
                ".jpeg": {"image/jpeg"},
                ".webp": {"image/webp"},
                ".tif": {"image/tiff"},
                ".tiff": {"image/tiff"},
            }[suffix]
            if format_ == "image"
            else declared in {"text/plain", "text/markdown"}
            if format_ in {"markdown", "text"}
            else declared in {"text/html", "application/xhtml+xml"}
        )
        if not compatible:
            raise ValueError("declared MIME type does not match source format")
    return format_


class TesseractOCR:
    """Local Tesseract adapter with orientation and bounded preprocessing."""

    def __init__(self, languages: str | None = None) -> None:
        self.languages = languages or os.getenv("THERAPY_OCR_LANGUAGES", "eng+spa+por")

    @property
    def metadata(self) -> dict[str, object]:
        import pytesseract

        requested = self.languages.split("+")
        installed = set(pytesseract.get_languages(config=""))
        missing = sorted(set(requested) - installed)
        if missing:
            raise RuntimeError(
                "Missing local Tesseract language packs: " + ", ".join(missing)
            )
        return {
            "engine": "tesseract",
            "version": str(pytesseract.get_tesseract_version()).splitlines()[0],
            "languages": requested,
        }

    @staticmethod
    def _prepare(image: Image) -> Image:
        import pytesseract
        from PIL import ImageOps

        prepared = ImageOps.exif_transpose(image).convert("RGB")
        if prepared.width * prepared.height > MAX_IMAGE_PIXELS:
            raise ValueError("image exceeds pixel limit")
        try:
            osd = pytesseract.image_to_osd(prepared)
            match = re.search(r"Rotate:\s*(90|180|270)", osd)
            if match:
                prepared = prepared.rotate(-int(match.group(1)), expand=True)
        except pytesseract.TesseractError:
            pass
        return ImageOps.autocontrast(ImageOps.grayscale(prepared))

    def recognize(self, image: Image) -> list[OCRBlock]:
        """Recognize line blocks with confidence and bounding boxes."""
        import pytesseract
        from pytesseract import Output

        _ = self.metadata
        data = pytesseract.image_to_data(
            self._prepare(image),
            lang=self.languages,
            config="--psm 6",
            output_type=Output.DICT,
        )
        lines: dict[tuple[int, int, int], list[int]] = {}
        for index, text in enumerate(data["text"]):
            if not str(text).strip():
                continue
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            lines.setdefault(key, []).append(index)
        result: list[OCRBlock] = []
        for indices in lines.values():
            text = " ".join(str(data["text"][index]).strip() for index in indices)
            scores = [max(0.0, float(data["conf"][index])) / 100 for index in indices]
            left = min(int(data["left"][index]) for index in indices)
            top = min(int(data["top"][index]) for index in indices)
            right = max(
                int(data["left"][index]) + int(data["width"][index])
                for index in indices
            )
            bottom = max(
                int(data["top"][index]) + int(data["height"][index])
                for index in indices
            )
            result.append(
                OCRBlock(text, sum(scores) / len(scores), (left, top, right, bottom))
            )
        return result


class _SafeHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignored_depth = 0
        self.current_tag: str | None = None
        self.current_heading: str | None = None
        self.buffer: list[str] = []
        self.blocks: list[tuple[str | None, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _IGNORED_HTML:
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"}:
            self._flush()
            self.current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED_HTML:
            self.ignored_depth = max(0, self.ignored_depth - 1)
            return
        if not self.ignored_depth and tag == self.current_tag:
            self._flush()
            self.current_tag = None

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and self.current_tag is not None:
            self.buffer.append(data)

    def _flush(self) -> None:
        text = _normalize(" ".join(self.buffer))
        self.buffer.clear()
        if not text:
            return
        if self.current_tag and self.current_tag.startswith("h"):
            self.current_heading = text
        else:
            self.blocks.append((self.current_heading, text))

    def close(self) -> None:
        self._flush()
        super().close()


def _text_blocks(text: str, *, markdown: bool) -> list[SourceBlock]:
    result: list[SourceBlock] = []
    heading: str | None = None
    section_counts: dict[str, int] = {}
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        body = _normalize(" ".join(paragraph))
        paragraph.clear()
        if not body:
            return
        section = _slug(heading or "document")
        section_counts[section] = section_counts.get(section, 0) + 1
        result.append(
            SourceBlock(
                anchor=f"section-{section}-block-{section_counts[section]}",
                text=body,
                extraction_method="digital",
                heading=heading,
                confidence=1.0,
            )
        )

    for line in text.splitlines():
        match = _HEADING_RE.match(line) if markdown else None
        if match:
            flush()
            heading = _normalize(match.group(2))
        elif line.strip():
            paragraph.append(line)
        else:
            flush()
    flush()
    return result


def _html_blocks(text: str) -> list[SourceBlock]:
    parser = _SafeHTMLParser()
    parser.feed(text)
    parser.close()
    counts: dict[str, int] = {}
    blocks: list[SourceBlock] = []
    for heading, body in parser.blocks:
        section = _slug(heading or "document")
        counts[section] = counts.get(section, 0) + 1
        blocks.append(
            SourceBlock(
                anchor=f"section-{section}-block-{counts[section]}",
                text=body,
                extraction_method="digital",
                heading=heading,
                confidence=1.0,
            )
        )
    return blocks


def _ocr_blocks(
    image: Image, backend: OCRBackend, *, prefix: str, page: int
) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    for index, item in enumerate(backend.recognize(image), start=1):
        text = _normalize(item.text)
        if not text:
            continue
        blocks.append(
            SourceBlock(
                anchor=f"{prefix}-block-{index}",
                text=text,
                extraction_method="ocr",
                page=page,
                confidence=item.confidence,
                needs_review=item.confidence < OCR_REVIEW_THRESHOLD,
                bbox=item.bbox,
            )
        )
    return blocks


def _pdf_blocks(data: bytes, backend: OCRBackend | None) -> list[SourceBlock]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PDF ingest requires the pymupdf dependency") from exc
    from PIL import Image

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("invalid or encrypted PDF") from exc
    blocks: list[SourceBlock] = []
    with document:
        if document.needs_pass:
            raise ValueError("encrypted PDF is not supported")
        if document.page_count > MAX_SOURCE_PAGES:
            raise ValueError("PDF exceeds page limit")
        for page_index, page in enumerate(document, start=1):
            digital = [
                _normalize(str(item[4]))
                for item in page.get_text("blocks")
                if len(item) >= 5 and _normalize(str(item[4]))
            ]
            if len(" ".join(digital)) >= 40:
                blocks.extend(
                    SourceBlock(
                        anchor=f"page-{page_index}-block-{index}",
                        text=text,
                        extraction_method="digital",
                        page=page_index,
                        confidence=1.0,
                    )
                    for index, text in enumerate(digital, start=1)
                )
                continue
            local_backend = backend or TesseractOCR()
            pixmap = page.get_pixmap(dpi=150, alpha=False, annots=False)
            if pixmap.width * pixmap.height > MAX_IMAGE_PIXELS:
                raise ValueError("rendered PDF page exceeds pixel limit")
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            blocks.extend(
                _ocr_blocks(
                    image, local_backend, prefix=f"page-{page_index}", page=page_index
                )
            )
    return blocks


def _image_blocks(data: bytes, backend: OCRBackend | None) -> list[SourceBlock]:
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("invalid image source") from exc
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("image exceeds pixel limit")
    return _ocr_blocks(image, backend or TesseractOCR(), prefix="image-1", page=1)


def _markdown(blocks: list[SourceBlock]) -> str:
    sections: list[str] = []
    last_page: int | None = None
    last_heading: str | None = None
    for block in blocks:
        if block.page is not None and block.page != last_page:
            sections.append(f"## Page {block.page}")
            last_page = block.page
        if block.heading and block.heading != last_heading:
            sections.append(f"### {block.heading}")
            last_heading = block.heading
        review = " [OCR REVIEW REQUIRED]" if block.needs_review else ""
        sections.append(f"<!-- anchor:{block.anchor} -->\n{block.text}{review}")
    return "\n\n".join(sections)


def extract_source(
    data: bytes,
    filename: str,
    declared_type: str | None = None,
    *,
    ocr_backend: OCRBackend | None = None,
) -> ExtractedSource:
    """Validate and normalize one supported local source without network access."""
    format_ = _validate_input(data, filename, declared_type)
    if format_ == "pdf":
        blocks = _pdf_blocks(data, ocr_backend)
        media_type = "application/pdf"
    elif format_ == "image":
        blocks = _image_blocks(data, ocr_backend)
        media_type = declared_type or "image/*"
    else:
        text = _decode_text(data)
        blocks = (
            _html_blocks(text)
            if format_ == "html"
            else _text_blocks(text, markdown=format_ == "markdown")
        )
        media_type = {
            "html": "text/html",
            "markdown": "text/markdown",
            "text": "text/plain",
        }[format_]
    if not blocks:
        raise ValueError("source contains no extractable text")
    if sum(len(block.text) for block in blocks) > MAX_SOURCE_CHARS:
        raise ValueError("extracted source exceeds character limit")
    used_ocr = any(block.extraction_method == "ocr" for block in blocks)
    metadata = (ocr_backend or TesseractOCR()).metadata if used_ocr else {}
    status = (
        "review_required" if any(block.needs_review for block in blocks) else "indexed"
    )
    return ExtractedSource(
        format_, media_type, _markdown(blocks), tuple(blocks), status, metadata
    )


__all__ = [
    "ExtractedSource",
    "MAX_SOURCE_BYTES",
    "OCRBackend",
    "OCRBlock",
    "SourceBlock",
    "TesseractOCR",
    "extract_source",
]
