"""FastAPI service.  Run:  uvicorn jobmatch.api:app --reload"""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .connectors import fetch_jobs
from .matcher import Matcher, load_sample_jobs
from .models import Job

app = FastAPI(title="ResumeJobMatch AI", version="1.0.0")
_matcher: Matcher | None = None
STATIC = Path(__file__).resolve().parents[1] / "static"


def get_matcher() -> Matcher:
    global _matcher
    if _matcher is None:
        _matcher = Matcher("auto")
    return _matcher


@app.get("/health")
def health():
    return {"status": "ok", "backend": get_matcher().backend_name}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/match")
async def match(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    location: str = Form(""),
    remote_only: bool = Form(False),
    live: bool = Form(False),
    top_k: int = Form(10),
):
    m = get_matcher()
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > 5_000_000:
            raise HTTPException(413, "File too large (max 5 MB)")
        try:
            profile = m.profile_from_file(data, file.filename)
        except Exception:
            raise HTTPException(422, "Could not read that file. Try PDF, DOCX or TXT, or paste the text.")
    elif text.strip():
        profile = m.profile(text)
    else:
        raise HTTPException(400, "Provide a resume file or text")
    if len(profile.skills) == 0:
        raise HTTPException(422, "No recognisable skills found in the resume text")

    pool: list[Job] = load_sample_jobs()
    sources: dict[str, str] = {}
    if live:
        query = " ".join([profile.roles[0][0]] + sorted(profile.skills, key=profile.skills.get, reverse=True)[:2]) \
            if profile.roles else ""
        live_jobs, sources = fetch_jobs(query, location)
        if live_jobs:
            pool = live_jobs          # rank real listings when we have them
    result = m.match(profile, pool, top_k=max(1, min(top_k, 30)), location=location, remote_only=remote_only)
    result["sources"] = sources or {"sample": "built-in synthetic jobs"}
    return result
