"""Resume parsing: text extraction (PDF/DOCX/TXT), sections, experience estimate, profile building."""
from __future__ import annotations
import io
import re
import zipfile
from datetime import date
from xml.etree import ElementTree

from .models import Profile
from .skills import extract_skills, infer_roles

SECTION_HEADERS = {
    "skills": r"(?:technical\s+)?skills|technologies|tech\s+stack|core\s+competencies",
    "experience": r"(?:work\s+|professional\s+)?experience|employment|internships?",
    "education": r"education|academics?|qualifications?",
    "projects": r"projects?|personal\s+projects",
    "summary": r"summary|objective|profile|about\s+me",
}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_EDU_LINE = re.compile(r"b\.?\s?tech|b\.?\s?e\b|m\.?\s?tech|bachelor|master|degree|university|college|school|"
                       r"cgpa|gpa|class\s+(?:x|xii)|hsc|ssc", re.I)


def extract_text(data: bytes, filename: str = "") -> str:
    name = filename.lower()
    if name.endswith(".pdf") or data[:4] == b"%PDF":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    if name.endswith(".docx") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paras = []
        for p in root.iter(f"{ns}p"):
            paras.append("".join(t.text or "" for t in p.iter(f"{ns}t")))
        return "\n".join(paras)
    return data.decode("utf-8", errors="ignore")


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        stripped = line.strip().strip(":").strip()
        hit = None
        if 0 < len(stripped) <= 40:
            for key, pat in SECTION_HEADERS.items():
                if re.fullmatch(pat, stripped, re.I):
                    hit = key
                    break
        if hit:
            current = hit
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


_RANGE = re.compile(
    r"(?:(?P<m1>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+)?(?P<y1>(?:19|20)\d\d)\s*"
    r"(?:-|–|—|to)\s*"
    r"(?:(?P<m2>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+)?"
    r"(?P<y2>(?:19|20)\d\d|present|current|now|ongoing)", re.I)


def estimate_years(text: str, today: date | None = None) -> float:
    """Union of employment date ranges (education lines ignored) plus explicit 'N years' claims."""
    today = today or date.today()
    spans: list[tuple[float, float]] = []
    for line in text.splitlines():
        if _EDU_LINE.search(line):
            continue
        for m in _RANGE.finditer(line):
            start = int(m["y1"]) + (_MONTHS[m["m1"].lower()[:3]] - 1) / 12 if m["m1"] else int(m["y1"])
            y2 = m["y2"].lower()
            if y2.isdigit():
                end = int(y2) + ((_MONTHS[m["m2"].lower()[:3]] - 1) / 12 if m["m2"] else 0.5)
            else:
                end = today.year + (today.month - 1) / 12
            if 0 <= end - start <= 40:
                spans.append((start, end))
    spans.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    claims = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:\w+\s+){0,2}experience", text, re.I)]
    return min(40.0, max(total, max(claims, default=0.0)))


def classify_level(years: float, text: str) -> str:
    low = text.lower()
    if years < 0.5 or (years < 1 and re.search(r"intern|fresher|student|graduate|trainee", low)):
        return "intern" if "intern" in low and years < 0.5 else "entry"
    if years < 2:
        return "entry" if years < 1 else "junior"
    if years < 4:
        return "junior"
    if years < 7:
        return "mid"
    return "senior"


def build_profile(text: str) -> Profile:
    text = re.sub(r"[ \t]+", " ", text).strip()
    sections = split_sections(text)
    skills = extract_skills(text, sections.get("skills", ""))
    years = estimate_years(sections.get("experience", "") or text)
    return Profile(
        text=text,
        skills=skills,
        roles=infer_roles(skills),
        years=years,
        level=classify_level(years, text),
        sections=sections,
    )
