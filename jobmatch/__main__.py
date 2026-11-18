"""CLI:  python -m jobmatch resume.pdf [--live] [--location Chennai] [--top 10] [--json]"""
import argparse
import json
import sys

from .connectors import fetch_jobs
from .matcher import Matcher, load_sample_jobs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobmatch", description="Match a resume to jobs")
    ap.add_argument("resume", help="path to .pdf, .docx or .txt")
    ap.add_argument("--live", action="store_true", help="fetch live listings (Remotive, Arbeitnow, RemoteOK, Adzuna)")
    ap.add_argument("--location", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--backend", default="auto", choices=["auto", "lsa", "transformer"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    m = Matcher(a.backend)
    with open(a.resume, "rb") as f:
        profile = m.profile_from_file(f.read(), a.resume)
    pool, status = load_sample_jobs(), {}
    if a.live:
        q = " ".join([profile.roles[0][0]] + sorted(profile.skills, key=profile.skills.get, reverse=True)[:2])
        live, status = fetch_jobs(q, a.location)
        pool = live or pool
    res = m.match(profile, pool, top_k=a.top, location=a.location)
    res["sources"] = status or {"sample": "built-in synthetic jobs"}
    if a.json:
        json.dump(res, sys.stdout, indent=2)
        return
    p = res["profile"]
    print(f"Backend: {res['backend']} | candidates: {res['candidates']} | sources: {res['sources']}")
    print(f"Level: {p['level']} ({p['years']} yrs) | roles: {', '.join(r['role'] for r in p['roles'][:3])}")
    print(f"Skills: {', '.join(p['skills'])}\n")
    for i, x in enumerate(res["matches"], 1):
        j = x["job"]
        print(f"{i:>2}. {x['score']:5.1f}%  {j['title']} @ {j['company']} ({j['location']})")
        print(f"      have: {', '.join(x['matched']) or '-'} | missing: {', '.join(x['missing_required']) or '-'}")
    if res["skills_to_learn"]:
        print("\nSkills that unlock the most of these jobs:",
              ", ".join(f"{g['skill']} ({g['jobs_requiring']})" for g in res["skills_to_learn"]))


if __name__ == "__main__":
    main()
