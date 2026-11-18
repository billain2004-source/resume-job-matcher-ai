"""Offline evaluation on synthetic resumes vs the synthetic job corpus.

Caveat: both sides are generated from the same taxonomy, so absolute numbers are optimistic. The point is a
sanity check and a fair *relative* comparison of ranking strategies (lexical / semantic / skills / hybrid /
learned weights). Real-world validation needs human-labelled resume-job pairs.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobmatch.embeddings import LSABackend  # noqa: E402
from jobmatch.matcher import load_sample_jobs  # noqa: E402
from jobmatch.parser import build_profile  # noqa: E402
from jobmatch.ranker import DEFAULT_WEIGHTS, FEATURES, Ranker  # noqa: E402
from jobmatch.skills import ADJACENT_ROLES, ROLE_PROFILES, TAXONOMY  # noqa: E402

BULLETS = ["Built and shipped features used by thousands of users", "Collaborated with a cross-functional team",
           "Improved reliability and reduced turnaround time", "Documented processes and mentored interns",
           "Presented results to stakeholders every sprint", "Automated repetitive manual work"]


def make_resume(role: str, rng: random.Random, noise: int = 3) -> str:
    core = ROLE_PROFILES[role]
    have = rng.sample(core, max(4, int(len(core) * rng.uniform(0.5, 0.8))))
    others = [s for r, p in ROLE_PROFILES.items() if r != role for s in p if s not in have]
    have += rng.sample(others, noise)
    def alias(s):
        opts = [s] + TAXONOMY.get(s, [])
        return rng.choice(opts)
    years = rng.choice([0, 1, 2, 4, 6])
    exp = f"Jan {2026 - years - 1} - Present" if years else "Jan 2026 - Present"
    return (f"Candidate\nSKILLS\n{', '.join(alias(s) for s in have)}\nEXPERIENCE\n{role}  {exp}\n- "
            + "\n- ".join(rng.sample(BULLETS, 3)) + "\n- Worked hands-on with " + ", ".join(alias(s) for s in rng.sample(have, 3)))


def grade(resume_role: str, job_role: str) -> int:
    if resume_role == job_role:
        return 2
    return 1 if job_role in ADJACENT_ROLES.get(resume_role, set()) else 0


def metrics(order_grades: list[int]) -> tuple[float, float, float]:
    p5 = sum(g == 2 for g in order_grades[:5]) / 5
    mrr = next((1 / (i + 1) for i, g in enumerate(order_grades) if g == 2), 0.0)
    dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(order_grades[:10]))
    ideal = sorted(order_grades, reverse=True)[:10]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal)) or 1.0
    return p5, mrr, dcg / idcg


def features_for(profile, jobs):
    r = Ranker(LSABackend())
    jobs, F, _ = r.features(profile, jobs)
    return jobs, F


def main(n_train=60, n_test=130, seed=1):
    jobs0 = load_sample_jobs()
    roles = list(ROLE_PROFILES)
    rng = random.Random(seed)
    train = [(role, make_resume(role, rng)) for role in (roles * 10)[:n_train]]
    test = [(role, make_resume(role, rng)) for role in (roles * 20)[:n_test]]

    X, y = [], []
    for role, txt in train:
        jobs, F = features_for(build_profile(txt), jobs0)
        X.append(F); y += [1 if grade(role, j.role_label) == 2 else 0 for j in jobs]
    X = np.vstack(X)
    lr = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced").fit(X, y)
    learned = {f: max(float(c), 0.0) for f, c in zip(FEATURES, lr.coef_[0])}
    if sum(learned.values()) == 0:
        learned = dict(DEFAULT_WEIGHTS)

    strategies = {
        "lexical only (TF-IDF)": {"lexical": 1, "semantic": 0, "skills": 0, "title": 0, "level": 0},
        "semantic only (LSA)": {"lexical": 0, "semantic": 1, "skills": 0, "title": 0, "level": 0},
        "skill coverage only": {"lexical": 0, "semantic": 0, "skills": 1, "title": 0, "level": 0},
        "hybrid (default weights)": DEFAULT_WEIGHTS,
        "hybrid (learned weights)": learned,
    }
    results = {}
    cache = []
    for role, txt in test:
        jobs, F = features_for(build_profile(txt), jobs0)
        cache.append((role, jobs, F))
    for name, w in strategies.items():
        wv = np.array([w[f] for f in FEATURES], float)
        ms = []
        for role, jobs, F in cache:
            order = np.argsort(-(F @ wv))
            ms.append(metrics([grade(role, jobs[i].role_label) for i in order]))
        results[name] = [float(x) for x in np.mean(ms, axis=0)]
    print(f"{n_test} held-out synthetic resumes, {len(jobs0)} jobs, {len(roles)} role families\n")
    print(f"{'strategy':28s} {'P@5':>6s} {'MRR':>6s} {'nDCG@10':>8s}")
    for k, (p, m, n) in results.items():
        print(f"{k:28s} {p:6.3f} {m:6.3f} {n:8.3f}")
    print("\nlearned weights:", {k: round(v, 3) for k, v in learned.items()})
    Path(__file__).with_name("eval_results.json").write_text(json.dumps(
        {"n_test": n_test, "n_jobs": len(jobs0), "results": results, "learned_weights": learned}, indent=1))


if __name__ == "__main__":
    main()
