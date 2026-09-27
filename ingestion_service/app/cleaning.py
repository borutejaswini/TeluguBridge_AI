from __future__ import annotations

import re
from collections import Counter
from math import ceil
from typing import Iterable

_PAGE_NUMBER = re.compile(r"^(?:[-–—] ?)?\d+(?: ?[-–—])?$")
_WHITESPACE = re.compile(r"[\t\u00a0 ]+")


def edge_lines(text: str, count: int = 3) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:count], lines[-count:]


def repeated_edge_lines(edge_pairs: Iterable[tuple[list[str], list[str]]], page_count: int) -> set[str]:
    if page_count < 3:
        return set()
    threshold = max(3, ceil(page_count * 0.5))
    counts: Counter[str] = Counter()
    for top, bottom in edge_pairs:
        counts.update({line.casefold() for line in top})
        counts.update({line.casefold() for line in bottom})
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_text(text: str, repeated_lines: set[str] | None = None) -> str:
    repeated = repeated_lines or set()
    kept: list[str] = []
    source_lines = text.replace("\x0c", "\n").splitlines()
    nonempty_positions = [i for i, line in enumerate(source_lines) if line.strip()]
    first_indices = set(nonempty_positions[:3])
    last_indices = set(nonempty_positions[-3:])
    edge_indices = first_indices | last_indices
    for index, raw_line in enumerate(source_lines):
        line = _WHITESPACE.sub(" ", raw_line).strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        folded = line.casefold()
        if index in edge_indices and folded in repeated:
            continue
        if index in edge_indices and _PAGE_NUMBER.fullmatch(line):
            continue
        kept.append(line)

    # Join visually wrapped prose lines conservatively. Keep headings, bullets,
    # numbered entries, and likely table rows on separate lines.
    paragraphs: list[str] = []
    current: list[str] = []
    for line in kept + [""]:
        if not line:
            if current:
                paragraphs.append(_join_prose_lines(current))
                current = []
            continue
        if _is_list_or_table_line(line):
            if current:
                paragraphs.append(_join_prose_lines(current))
                current = []
            paragraphs.append(line)
        else:
            current.append(line)
    return "\n\n".join(p for p in paragraphs if p).strip()


def _is_list_or_table_line(line: str) -> bool:
    return bool(
        re.match(r"^(?:[•●▪◦*-]\s+|\d+[.)]\s+|\(?[a-zA-Z]\)\s+)", line)
        or "\t" in line
    )


def _join_prose_lines(lines: list[str]) -> str:
    result: list[str] = []
    for line in lines:
        if result and _looks_heading(result[-1]):
            result.append("\n")
        elif result:
            result.append(" ")
        result.append(line)
    return "".join(result).replace("\n ", "\n").strip()


def _looks_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 100 or re.search(r"[.!?;:]$", text):
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+|chapter\s+|section\s+)", text, re.I):
        return True
    if text.casefold().startswith(("pest:", "symptoms:", "management:", "favorable factors:")):
        return True
    letters = [char for char in text if char.isalpha()]
    if letters and (text.isupper() or sum(c.isupper() for c in letters) / len(letters) > 0.72):
        return True
    words = text.split()
    return bool(
        letters
        and 1 <= len(words) <= 10
        and all(
            not any(char.isalpha() for char in word)
            or next(char for char in word if char.isalpha()).isupper()
            for word in words
        )
    )
