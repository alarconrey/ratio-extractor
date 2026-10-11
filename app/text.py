"""Paragraph splitting, section detection and word counting.

The dataset builder and the API use these same functions, so the model sees the
same paragraphs in production that it saw during training.
"""

import re
import unicodedata

from bs4 import BeautifulSoup

HTML_TAG = re.compile(r"<\s*(p|li|h[1-4]|div|br|span|b|table)\b", re.IGNORECASE)
NUMBERED_HEADING = re.compile(
    r"^\s*(?:[IVX]{1,4}|\d{1,2})(?:\.\d{1,2})*\s*[\.\-–)]+\s*([A-Za-zÁÉÍÓÚÑáéíóúñ][^\.:]{3,80}?)(?:\s*[\.:]-?|\s*$)",
    re.IGNORECASE,
)
KEYWORD_HEADING = re.compile(r"^\s*(POR TANTO|VISTOS|CONSIDERANDO|RESUELVE|FALLA)\b", re.IGNORECASE)
NOT_A_HEADING_START = ("que", "el ", "la ", "los ", "las ", "en ", "de ", "por ", "con ", "se ")
SECTION_KEYWORDS = [
    ("fundament", "fundamentos"), ("por tanto", "por_tanto"), ("resuelve", "por_tanto"), ("falla", "por_tanto"),
    ("antecedent", "antecedentes"), ("argument", "argumentos"), ("recurso", "argumentos"), ("demanda", "argumentos"),
    ("contestaci", "argumentos"), ("problema", "problema_juridico"), ("consider", "fundamentos"),
    ("vistos", "encabezado"), ("análisis", "fundamentos"), ("analisis", "fundamentos"),
]
SECTIONS = ["encabezado", "antecedentes", "argumentos", "problema_juridico", "fundamentos", "por_tanto", "otro"]


def normalize_space(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def words(text):
    """Lowercase word tokens; quotes and periods are ignored."""
    return re.findall(r"\w+", re.sub(r"[“”\"«»‘’'…\.]", " ", text.lower()))


def word_ngrams(tokens, size=8):
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def looks_like_html(text):
    return bool(HTML_TAG.search(text or ""))


def split_paragraphs(content):
    """Split an HTML resolution (as stored in the database) or plain text into paragraphs."""
    if looks_like_html(content):
        soup = BeautifulSoup(content or "", "lxml")
        paragraphs = []
        for element in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
            if element.find(["p", "li"]):
                continue  # Keep only innermost blocks to avoid duplicated text.
            text = normalize_space(element.get_text(" "))
            if len(text) >= 3:
                paragraphs.append(text)
        return paragraphs
    # Plain text: one paragraph per non-empty line.
    return [normalize_space(line) for line in (content or "").splitlines() if len(normalize_space(line)) >= 3]


def section_category(heading):
    heading = heading.lower()
    for keyword, category in SECTION_KEYWORDS:
        if keyword in heading:
            return category
    return "otro"


def detect_heading(paragraph):
    """Return the heading text if the paragraph opens a new section, else None."""
    keyword = KEYWORD_HEADING.match(paragraph)
    if keyword:
        return keyword.group(1)
    numbered = NUMBERED_HEADING.match(paragraph)
    if numbered:
        title = numbered.group(1).strip()
        if len(title.split()) <= 12 and len(paragraph.split()) <= 30 and not title.lower().startswith(NOT_A_HEADING_START):
            return title
    return None


def paragraph_features(paragraphs):
    """Position, section and length for each paragraph, in document order."""
    rows = []
    section, heading_text = "encabezado", ""
    last = max(len(paragraphs) - 1, 1)
    for index, paragraph in enumerate(paragraphs):
        heading = detect_heading(paragraph)
        if heading:
            section, heading_text = section_category(heading), heading
        rows.append({
            "idx": index, "n_parrafos": len(paragraphs), "pos_rel": round(index / last, 3),
            "seccion": section, "encabezado": heading_text[:60], "n_palabras": len(words(paragraph)), "texto": paragraph,
        })
    return rows
