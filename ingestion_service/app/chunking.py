from __future__ import annotations

import hashlib
import re

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+(?=[^\s])")
_KNOWN_SECTION = re.compile(
    r"^(?:pest|disease|symptoms?|management|control|identification|biology|life cycle|"
    r"favourable factors|favorable factors|economic threshold|etl|integrated pest management)\s*:?(?:\s*\([^)]*\))?$",
    re.I,
)


def _is_parent_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 100 or re.search(r"[.!?;:]$", text):
        return False
    # Explicit record headings name a new pest/disease section. Inline metadata
    # labels (Pest Name, Target Crop, Symptoms, etc.) are content under that
    # section and must never replace the parent entity.
    if re.match(r"^(?:pest|disease)\s+record\s+\d+\s*:\s*\S", text, re.I):
        return True
    if text.startswith(("�", "•", "●", "▪", "◦", "-", "*")):
        return False
    if ":" in text:
        return False
    numbered_heading = re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z]", text)
    if text[0].isdigit() and not numbered_heading:
        return False
    if numbered_heading or re.match(r"^(?:chapter\s+|section\s+)", text, re.I):
        return True
    words = text.split()
    letters = [char for char in text if char.isalpha()]
    # Title-case/all-caps short standalone labels such as "Stem Borer".
    return bool(letters and 1 <= len(words) <= 10 and all(
        not any(char.isalpha() for char in word) or next(char for char in word if char.isalpha()).isupper()
        for word in words
    ))


def _is_child_heading(line: str) -> bool:
    return bool(_KNOWN_SECTION.fullmatch(line.strip()))


def _split_oversized(paragraph: str, target: int) -> list[str]:
    if len(paragraph) <= target:
        return [paragraph]
    sentences = [part.strip() for part in _SENTENCE_END.split(paragraph) if part.strip()]
    if len(sentences) <= 1:
        # An indivisible sentence may exceed the target; keeping it intact is
        # safer than cutting a dosage, chemical name, or instruction mid-span.
        return [paragraph]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > target:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def _pack_section(
    parent: str | None,
    child: str | None,
    paragraphs: list[str],
    target_chars: int,
    include_parent_in_text: bool,
) -> list[tuple[str | None, str]]:
    if not paragraphs:
        return []
    section = f"{parent} > {child}" if parent and child else (child or parent)
    heading_lines = ([parent] if parent and include_parent_in_text else []) + ([child] if child else [])
    prefix = "\n".join(heading_lines)
    output: list[tuple[str | None, str]] = []
    current: list[str] = []

    def emit() -> None:
        if current:
            body = "\n\n".join(current)
            text = f"{prefix}\n{body}" if prefix else body
            output.append((section, text.strip()))
            current.clear()

    for paragraph in paragraphs:
        for part in _split_oversized(paragraph, max(1, target_chars - len(prefix) - 1)):
            candidate = "\n\n".join([*current, part])
            if current and len(prefix) + 1 + len(candidate) > target_chars:
                emit()
            current.append(part)
    emit()
    return output


def chunk_page(
    text: str,
    target_chars: int,
    *,
    inherited_section: str | None = None,
) -> list[tuple[str | None, str]]:
    """Chunk by heading hierarchy and paragraphs, retaining parent context.

    A recognized child section is its own coherent chunk group even when the
    combined section would fit below the target. If a section needs further
    splitting, each continuation repeats its parent/child heading metadata.
    """
    if target_chars < 1:
        raise ValueError("target_chars must be positive")

    output: list[tuple[str | None, str]] = []
    parent: str | None = inherited_section
    parent_is_inherited = inherited_section is not None
    child: str | None = None
    section_paragraphs: list[str] = []
    pending_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal pending_lines
        if pending_lines:
            section_paragraphs.append(" ".join(line.strip() for line in pending_lines).strip())
            pending_lines = []

    def flush_section() -> None:
        nonlocal section_paragraphs
        flush_paragraph()
        output.extend(
            _pack_section(
                parent,
                child,
                section_paragraphs,
                target_chars,
                include_parent_in_text=not parent_is_inherited,
            )
        )
        section_paragraphs = []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if _is_child_heading(line):
            flush_section()
            child = line.rstrip(":")
            continue
        if _is_parent_heading(line):
            flush_section()
            parent = line
            parent_is_inherited = False
            child = None
            continue
        pending_lines.append(line)
    flush_section()
    return output


def make_chunk_id(document_id: str, page_number: int, chunk_index: int, section: str | None, text: str) -> str:
    # Page and sequential position distinguish repeated text while all inputs
    # remain stable for repeated ingestion of the same source/configuration.
    payload = "\0".join((document_id, str(page_number), str(chunk_index), section or "", text))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{document_id[:12]}-p{page_number:04d}-c{chunk_index:03d}-{digest}"
