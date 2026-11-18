"""Live job-board connectors. Each returns normalised Job objects and never raises on network errors.

Sources
  remotive   free public API, remote jobs                      (no key)
  arbeitnow  free public API, Europe + remote                  (no key)
  remoteok   free public API, remote jobs                      (no key)
  adzuna     aggregator with India/UK/US/... coverage          (ADZUNA_APP_ID + ADZUNA_APP_KEY)

Scraping LinkedIn/Naukri/Indeed is against their terms and is intentionally not done here.
"""
from __future__ import annotations
import html
import os
import re
from typing import Callable

import httpx

from .models import Job

UA = {"User-Agent": "ResumeJobMatchAI/1.0 (+https://github.com/)"}


def clean_html(raw: str) -> str:
    text = re.sub(r"<(script|style).*?</\1>", " ", raw or "", flags=re.S | re.I)
    text = re.sub(r"</(p|div|li|ul|ol|h\d|br)>|<br\s*/?>", ".\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", html.unescape(text)).strip()


def _remotive(client: httpx.Client, query: str, location: str, limit: int) -> list[Job]:
    r = client.get("https://remotive.com/api/remote-jobs", params={"search": query, "limit": limit})
    r.raise_for_status()
    return [Job(id=f"remotive-{d['id']}", title=d["title"], company=d.get("company_name", ""),
                description=clean_html(d.get("description", ""))[:6000],
                location=d.get("candidate_required_location") or "Remote", remote=True,
                url=d.get("url", ""), source="remotive")
            for d in r.json().get("jobs", [])]


def _arbeitnow(client: httpx.Client, query: str, location: str, limit: int) -> list[Job]:
    r = client.get("https://www.arbeitnow.com/api/job-board-api")
    r.raise_for_status()
    words = {w for w in re.findall(r"[a-z+#.]+", query.lower()) if len(w) > 2}
    out = []
    for d in r.json().get("data", []):
        blob = f"{d.get('title', '')} {' '.join(d.get('tags', []))} {d.get('description', '')[:1500]}".lower()
        if words and not any(w in blob for w in words):
            continue
        out.append(Job(id=f"arbeitnow-{d['slug']}", title=d["title"], company=d.get("company_name", ""),
                       description=clean_html(d.get("description", ""))[:6000], location=d.get("location", ""),
                       remote=bool(d.get("remote")), url=d.get("url", ""), source="arbeitnow"))
        if len(out) >= limit:
            break
    return out


def _remoteok(client: httpx.Client, query: str, location: str, limit: int) -> list[Job]:
    r = client.get("https://remoteok.com/api", headers=UA)
    r.raise_for_status()
    words = {w for w in re.findall(r"[a-z+#.]+", query.lower()) if len(w) > 2}
    out = []
    for d in r.json():
        if not isinstance(d, dict) or "position" not in d:
            continue
        blob = f"{d['position']} {' '.join(d.get('tags', []))}".lower()
        if words and not any(w in blob for w in words):
            continue
        out.append(Job(id=f"remoteok-{d.get('id')}", title=d["position"], company=d.get("company", ""),
                       description=clean_html(d.get("description", ""))[:6000],
                       location=d.get("location") or "Remote", remote=True,
                       url=d.get("url") or d.get("apply_url", ""), source="remoteok"))
        if len(out) >= limit:
            break
    return out


def _adzuna(client: httpx.Client, query: str, location: str, limit: int, country: str = "in") -> list[Job]:
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        raise RuntimeError("ADZUNA_APP_ID / ADZUNA_APP_KEY not set")
    params = {"app_id": app_id, "app_key": app_key, "what": query, "results_per_page": min(limit, 50),
              "content-type": "application/json"}
    if location:
        params["where"] = location
    r = client.get(f"https://api.adzuna.com/v1/api/jobs/{os.getenv('ADZUNA_COUNTRY', country)}/search/1",
                   params=params)
    r.raise_for_status()
    return [Job(id=f"adzuna-{d['id']}", title=clean_html(d["title"]), company=d.get("company", {}).get("display_name", ""),
                description=clean_html(d.get("description", "")), location=d.get("location", {}).get("display_name", ""),
                remote="remote" in d.get("title", "").lower(), url=d.get("redirect_url", ""), source="adzuna")
            for d in r.json().get("results", [])]


CONNECTORS: dict[str, Callable[..., list[Job]]] = {
    "remotive": _remotive, "arbeitnow": _arbeitnow, "remoteok": _remoteok, "adzuna": _adzuna,
}


def fetch_jobs(query: str, location: str = "", sources: list[str] | None = None, limit: int = 30,
               client: httpx.Client | None = None) -> tuple[list[Job], dict[str, str]]:
    """Fetch from several sources, de-duplicate on (title, company). Returns (jobs, per-source status)."""
    sources = sources or ["remotive", "arbeitnow", "remoteok"] + (["adzuna"] if os.getenv("ADZUNA_APP_ID") else [])
    own = client is None
    client = client or httpx.Client(timeout=12, follow_redirects=True, headers=UA)
    jobs: list[Job] = []
    status: dict[str, str] = {}
    try:
        for name in sources:
            fn = CONNECTORS.get(name)
            if not fn:
                status[name] = "unknown source"
                continue
            try:
                got = fn(client, query, location, limit)
                jobs += got
                status[name] = f"ok ({len(got)})"
            except Exception as e:  # network down, rate limit, bad JSON: report, don't crash
                status[name] = f"failed: {type(e).__name__}"
    finally:
        if own:
            client.close()
    seen, unique = set(), []
    for j in jobs:
        key = (j.title.lower().strip(), j.company.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(j)
    return unique, status
