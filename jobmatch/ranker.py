"""Hybrid, explainable ranker.

final = location_factor * sum_i w_i * feature_i  with features
  semantic : embedding cosine between resume and job text
  lexical  : TF-IDF (1-2 gram) cosine
  skills   : IDF-weighted coverage of required (+0.5x preferred) skills, partial credit for related skills
  title    : whether the job's inferred role matches the candidate's top inferred roles
  level    : experience fit (penalises big gaps and large over-qualification)
Weights are hand-set defaults and can be learned from labelled pairs (see evaluate.py).
"""
from __future__ import annotations
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import EmbeddingBackend, LSABackend
from .models import Job, Match, Profile
from .skills import find_skills, infer_roles, related, role_specificity

FEATURES = ["semantic", "lexical", "skills", "title", "level"]
DEFAULT_WEIGHTS = {"semantic": 0.25, "lexical": 0.10, "skills": 0.40, "title": 0.15, "level": 0.10}

_LEVEL_YEARS = [(r"\b(intern|internship|trainee)\b", 0.0), (r"\b(entry|junior|jr|graduate|fresher)\b", 0.0),
                (r"\b(senior|sr|lead|principal|staff)\b", 5.0), (r"\b(mid|intermediate)\b", 3.0)]


def parse_job_requirements(job: Job) -> Job:
    """Fill required/preferred skills and min_years from free text when the source didn't provide them."""
    if not job.required and not job.preferred:
        req, pref, mode = Counter(), Counter(), "req"
        for line in re.split(r"[\n.•;]+", job.description):
            low = line.lower()
            if re.search(r"nice to have|preferred|bonus|a plus|good to have|optional", low):
                mode = "pref"
            elif re.search(r"requirements?|qualifications?|must have|you have|responsibilities|what you", low):
                mode = "req"
            (pref if mode == "pref" else req).update(find_skills(line))
        job.required = [s for s in req]
        job.preferred = [s for s in pref if s not in req]
        if not job.required and not job.preferred:   # fall back to anything mentioned in the title
            job.required = list(find_skills(job.title))
    if job.min_years is None:
        m = re.search(r"(\d+)\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)", job.description, re.I)
        if m:
            job.min_years = float(m.group(1))
        else:
            job.min_years = next((y for pat, y in _LEVEL_YEARS if re.search(pat, job.title, re.I)), 1.0)
            if re.search(r"intern|trainee|junior|entry|fresher|graduate", job.title, re.I):
                job.min_years = 0.0
    return job


def _level_fit(years: float, job_min: float) -> float:
    if years + 0.5 >= job_min:
        over = years - (job_min + 4)
        return 1.0 if over <= 0 else max(0.6, 1 - over * 0.08)   # mild penalty if far over-qualified
    return max(0.0, 1 - (job_min - years - 0.5) / 3.0)


class Ranker:
    def __init__(self, backend: EmbeddingBackend | None = None, weights: dict[str, float] | None = None):
        self.backend = backend or LSABackend()
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    # ---- features -------------------------------------------------------------------------
    def features(self, profile: Profile, jobs: list[Job]):
        jobs = [parse_job_requirements(j) for j in jobs]
        job_texts = [j.text() for j in jobs]
        resume_text = profile.text[:8000]

        if self.backend.needs_fit:
            self.backend.fit(job_texts + [resume_text])
        E = self.backend.encode(job_texts)
        r = self.backend.encode([resume_text])[0]
        sem = np.clip(E @ r, 0, None)

        tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                             token_pattern=r"(?u)\b[\w+#.]+\b")
        X = tf.fit_transform(job_texts + [resume_text])
        lex = np.asarray((X[:-1] @ X[-1].T).todense()).ravel()

        # scale the two text similarities to the best candidate so they sit on a 0..1 scale
        sem = sem / max(sem.max(), 1e-9)
        lex = lex / max(lex.max(), 1e-9)

        # skill IDF over this candidate pool
        df = Counter(s for j in jobs for s in set(j.required) | set(j.preferred))
        n = len(jobs)
        # rarer-in-pool skills and role-defining skills (wordpress) count more than generic ones (html, git)
        idf = lambda s: (math.log((n + 1) / (df.get(s, 0) + 1)) + 1.0) * (1.0 + 2.0 * role_specificity(s))
        have = set(profile.skills)
        top_roles = [r for r, _ in profile.roles[:2]]
        wide_roles = [r for r, _ in profile.roles[:4]]

        rows, details = [], []
        for j, s_sem, s_lex in zip(jobs, sem, lex):
            num = den = 0.0
            matched, transfer, miss_req, miss_pref = [], [], [], []
            for group, mult in ((j.required, 1.0), (j.preferred, 0.5)):
                for sk in group:
                    w = idf(sk) * mult
                    den += w
                    if sk in have:
                        num += w
                        matched.append(sk)
                    elif (alt := related(sk, have)):
                        num += 0.5 * w
                        transfer.append((sk, alt))
                    else:
                        (miss_req if group is j.required else miss_pref).append(sk)
            skill = num / den if den else 0.0

            job_roles = [r for r, _ in infer_roles(set(j.required) | set(j.preferred) | set(find_skills(j.title)), 2)]
            if job_roles and job_roles[0] in top_roles:
                title = 1.0
            elif any(r in wide_roles for r in job_roles):
                title = 0.5
            else:
                title = 0.0

            level = _level_fit(profile.years, j.min_years or 0.0)
            rows.append([s_sem, s_lex, skill, title, level])
            details.append((matched, transfer, miss_req, miss_pref))
        return jobs, np.array(rows), details

    # ---- ranking --------------------------------------------------------------------------
    def rank(self, profile: Profile, jobs: list[Job], top_k: int = 10, location: str = "",
             remote_only: bool = False) -> list[Match]:
        if remote_only:
            jobs = [j for j in jobs if j.remote]
        if not jobs:
            return []
        jobs, F, details = self.features(profile, jobs)
        w = np.array([self.weights[f] for f in FEATURES])
        scores = F @ w / w.sum()
        loc = location.strip().lower()
        out = []
        for job, s, row, (matched, transfer, mreq, mpref) in zip(jobs, scores, F, details):
            if loc:
                factor = 1.0 if (job.remote or loc in job.location.lower()) else 0.85
                s *= factor
            out.append(Match(job=job, score=float(s), components=dict(zip(FEATURES, row.tolist())),
                             matched=matched, transferable=transfer, missing_required=mreq,
                             missing_preferred=mpref))
        out.sort(key=lambda m: m.score, reverse=True)
        return out[:top_k]


def skill_gap_report(matches: list[Match], top: int = 8) -> list[dict]:
    """Which missing skills would unlock the most of the shortlisted jobs."""
    c = Counter(s for m in matches for s in m.missing_required)
    return [{"skill": s, "jobs_requiring": n} for s, n in c.most_common(top)]
