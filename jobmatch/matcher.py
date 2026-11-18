"""High-level orchestration: resume -> profile -> (fetch jobs) -> rank -> gap report."""
from __future__ import annotations
import json
from pathlib import Path

from .embeddings import EmbeddingBackend, get_backend
from .models import Job, Profile
from .parser import build_profile, extract_text
from .ranker import Ranker, skill_gap_report

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_jobs.json"


def load_sample_jobs() -> list[Job]:
    return [Job(**d) for d in json.loads(SAMPLE_PATH.read_text())]


class Matcher:
    def __init__(self, backend: str | EmbeddingBackend = "auto", weights: dict | None = None):
        be = get_backend(backend) if isinstance(backend, str) else backend
        self.ranker = Ranker(be, weights)

    @property
    def backend_name(self) -> str:
        return self.ranker.backend.name

    def profile(self, text: str) -> Profile:
        return build_profile(text)

    def profile_from_file(self, data: bytes, filename: str = "") -> Profile:
        return build_profile(extract_text(data, filename))

    def match(self, resume: str | Profile, jobs: list[Job] | None = None, top_k: int = 10,
              location: str = "", remote_only: bool = False) -> dict:
        profile = resume if isinstance(resume, Profile) else build_profile(resume)
        pool = jobs if jobs is not None else load_sample_jobs()
        matches = self.ranker.rank(profile, pool, top_k=top_k, location=location, remote_only=remote_only)
        return {
            "profile": profile.to_dict(),
            "backend": self.backend_name,
            "candidates": len(pool),
            "matches": [m.to_dict() for m in matches],
            "skills_to_learn": skill_gap_report(matches),
        }
