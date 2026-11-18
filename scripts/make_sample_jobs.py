"""Generate a synthetic job corpus (offline demo + evaluation). Real listings come from connectors.py."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobmatch.skills import ROLE_PROFILES  # noqa: E402

RESPONSIBILITIES = {
    "Machine Learning Engineer": ["Train, evaluate and deploy machine learning models to production",
                                  "Build data pipelines and feature stores for model training",
                                  "Monitor model drift and retrain models automatically"],
    "Data Scientist": ["Run experiments and statistical analyses to guide product decisions",
                       "Build predictive models and communicate findings to stakeholders",
                       "Design A/B tests and interpret results"],
    "Data Analyst": ["Build dashboards and recurring reports for business teams",
                     "Write queries to clean, join and analyse large datasets",
                     "Translate business questions into metrics"],
    "Frontend Developer": ["Build responsive, accessible user interfaces and reusable components",
                           "Optimise page performance and Core Web Vitals",
                           "Collaborate with designers to ship polished features"],
    "Backend Developer": ["Design and build scalable REST services and data models",
                          "Own database performance, caching and reliability",
                          "Write automated tests and review code"],
    "Full Stack Developer": ["Build features end to end across the web client and the server API",
                             "Design database schemas and integrate third-party services",
                             "Ship quickly in a small product team"],
    "DevOps Engineer": ["Automate build, test and deployment pipelines",
                        "Manage cloud infrastructure as code and container orchestration",
                        "Set up monitoring, alerting and incident response"],
    "Mobile App Developer": ["Build and release cross-platform and native mobile applications",
                             "Integrate APIs, push notifications and analytics",
                             "Profile and improve app performance and crash rates"],
    "QA Automation Engineer": ["Design and maintain automated end to end test suites",
                               "Plan test strategies and triage defects with developers",
                               "Integrate tests into CI pipelines"],
    "Cybersecurity Analyst": ["Monitor alerts, investigate incidents and respond to threats",
                              "Run vulnerability scans and penetration tests",
                              "Harden systems and document security controls"],
    "Customer Success Specialist": ["Resolve customer tickets and own onboarding for new accounts",
                                    "Review leads and flag fraudulent signups",
                                    "Track account health in the CRM and reduce churn"],
    "WordPress Developer": ["Build and customise WordPress themes and plugins",
                            "Maintain marketing sites and improve page speed and SEO",
                            "Migrate content and manage hosting"],
    "Digital Marketing Specialist": ["Plan and run email and paid campaigns",
                                     "Grow organic traffic through SEO and content",
                                     "Report on funnel metrics and lead quality"],
}
SHORT = {"Machine Learning Engineer": "ML Engineer", "Data Scientist": "Data Scientist",
         "Data Analyst": "Data Analyst", "Frontend Developer": "Frontend Developer",
         "Backend Developer": "Backend Engineer", "Full Stack Developer": "Full Stack Developer",
         "DevOps Engineer": "DevOps Engineer", "Mobile App Developer": "Mobile Developer",
         "QA Automation Engineer": "QA Automation Engineer", "Cybersecurity Analyst": "Security Analyst",
         "Customer Success Specialist": "Customer Success Associate", "WordPress Developer": "WordPress Developer",
         "Digital Marketing Specialist": "Digital Marketing Executive"}
LEVELS = [("Intern", 0), ("Junior", 1), ("", 3), ("Senior", 5), ("Associate", 0)]
COMPANIES = ["NimbusWorks", "Fernleaf Labs", "Quartz Systems", "Brightline", "Kestrel Tech", "Orchid Analytics",
             "Pinegrove", "Helio Cloud", "Marlow & Co", "Zenith Soft", "Lumen Data", "Atlas Digital"]
CITIES = ["Chennai", "Bengaluru", "Hyderabad", "Pune", "Mumbai", "Delhi NCR"]


def make(seed: int = 7, per_role: int = 6):
    rng = random.Random(seed)
    jobs, n = [], 0
    for role, core in ROLE_PROFILES.items():
        for i in range(per_role):
            lvl, yrs = LEVELS[i % len(LEVELS)]
            k = rng.randint(5, 7)
            req = rng.sample(core, min(k, len(core)))
            rest = [s for s in core if s not in req]
            pref = rng.sample(rest, min(rng.randint(2, 3), len(rest)))
            title = f"{lvl} {SHORT[role]}".strip()
            remote = rng.random() < 0.3
            exp = f"{yrs}+ years of experience." if yrs >= 1 else "Freshers and recent graduates welcome."
            resp = ". ".join(RESPONSIBILITIES[role])
            desc = (f"Responsibilities: {resp}. Requirements: {exp} Hands-on experience with "
                    f"{', '.join(req)}. Nice to have: {', '.join(pref)}.")
            n += 1
            jobs.append({
                "id": f"sample-{n:03d}", "title": title, "company": rng.choice(COMPANIES),
                "description": desc, "location": "Remote" if remote else rng.choice(CITIES),
                "remote": remote, "url": f"https://example.com/sample-jobs/{n:03d}", "source": "sample",
                "required": req, "preferred": pref, "min_years": float(yrs),
                "role_label": role,
            })
    return jobs


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "jobmatch" / "data" / "sample_jobs.json"
    jobs = make()
    out.write_text(json.dumps(jobs, indent=1))
    print(f"wrote {len(jobs)} jobs -> {out}")
