import json
from datetime import date

import httpx
from fastapi.testclient import TestClient

from jobmatch import Matcher
from jobmatch.api import app
from jobmatch.connectors import clean_html, fetch_jobs
from jobmatch.models import Job
from jobmatch.parser import build_profile, estimate_years, extract_text, split_sections
from jobmatch.ranker import parse_job_requirements
from jobmatch.skills import find_skills, infer_roles, related

ML_RESUME = """SKILLS
Python, PyTorch, sklearn, pandas, NumPy, Docker, MLflow, NLP
EXPERIENCE
ML Engineer, Acme  Jan 2021 - Dec 2023
- Trained deep learning models and deployed with MLOps pipelines on AWS
"""


def test_alias_normalisation():
    c = find_skills("k8s, Postgres, sklearn, Node.js with Express, C++ and C#, ReactJS")
    for s in ["kubernetes", "postgresql", "scikit-learn", "node.js", "express", "c++", "c#", "react"]:
        assert s in c


def test_common_words_not_skills():
    assert "express" not in find_skills("We express interest in candidates")
    assert "rest api" not in find_skills("take a rest")


def test_role_inference():
    roles = infer_roles(find_skills(ML_RESUME))
    assert roles[0][0] == "Machine Learning Engineer"


def test_related_partial_credit():
    assert related("tensorflow", {"pytorch"}) == "pytorch"
    assert related("tensorflow", {"excel"}) is None


def test_years_union_and_education_ignored():
    txt = "B.Tech 2018 - 2022\nDev A  Jan 2020 - Dec 2021\nDev B  Jun 2021 - Jun 2022"
    y = estimate_years(txt, today=date(2026, 1, 1))
    assert 2.4 < y < 3.1          # union of 2020..2022.5, education range excluded


def test_sections_split():
    s = split_sections("Name\nSKILLS\nPython\nEDUCATION\nB.Tech")
    assert "Python" in s["skills"] and "B.Tech" in s["education"]


def test_extract_text_txt_and_docx_roundtrip():
    assert extract_text(b"hello", "a.txt") == "hello"
    import io, zipfile
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
           '<w:p><w:r><w:t>Python dev</w:t></w:r></w:p></w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    assert extract_text(buf.getvalue(), "r.docx").strip() == "Python dev"


def test_job_requirement_parsing():
    j = Job("1", "Backend Engineer", "X", "Requirements: 3+ years with Python and Django. Nice to have: Kafka.")
    parse_job_requirements(j)
    assert "python" in j.required and "django" in j.required and j.preferred == ["kafka"] and j.min_years == 3


def test_ml_resume_ranks_ml_jobs_first():
    r = Matcher("lsa").match(ML_RESUME, top_k=5)
    assert r["matches"][0]["job"]["role_label"] == "Machine Learning Engineer"
    assert all(a["score"] >= b["score"] for a, b in zip(r["matches"], r["matches"][1:]))


def test_senior_job_penalised_for_fresher():
    fresher = "SKILLS\nPython, pandas, scikit-learn, machine learning\nEXPERIENCE\nIntern  Jan 2026 - Present"
    r = Matcher("lsa").match(fresher, top_k=10)
    top = r["matches"][0]["job"]
    assert not top["title"].startswith("Senior")


def test_location_and_remote_filters():
    r = Matcher("lsa").match(ML_RESUME, top_k=30, remote_only=True)
    assert r["matches"] and all(m["job"]["remote"] for m in r["matches"])


def test_clean_html():
    assert "bold" in clean_html("<p>Some <b>bold</b> &amp; text</p>") and "<" not in clean_html("<p>x</p>")


def test_connectors_with_mock_transport():
    def handler(req: httpx.Request):
        if "remotive" in req.url.host:
            return httpx.Response(200, json={"jobs": [{"id": 1, "title": "Python Developer", "company_name": "A",
                                  "description": "<p>Python and Django</p>", "url": "u", "candidate_required_location": "Worldwide"}]})
        if "arbeitnow" in req.url.host:
            return httpx.Response(200, json={"data": [
                {"slug": "s", "title": "Python Developer", "company_name": "A", "description": "x", "remote": True, "url": "u2", "tags": []}]})
        return httpx.Response(500)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    jobs, status = fetch_jobs("python", sources=["remotive", "arbeitnow", "remoteok"], client=client)
    assert len(jobs) == 1                         # duplicate (title, company) removed
    assert status["remoteok"].startswith("failed") and status["remotive"].startswith("ok")


def test_api_text_and_errors():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/api/match", data={"text": ML_RESUME, "top_k": 3})
    assert r.status_code == 200 and len(r.json()["matches"]) == 3
    assert c.post("/api/match", data={"text": ""}).status_code == 400
    assert c.post("/api/match", data={"text": "no tech words at all in this resume text sample"}).status_code == 422
    up = c.post("/api/match", files={"file": ("cv.txt", ML_RESUME.encode())}, data={"top_k": 2})
    assert up.status_code == 200
