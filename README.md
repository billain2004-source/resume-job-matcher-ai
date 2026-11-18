# ResumeJobMatch AI

Upload a resume (PDF, DOCX or TXT) and get ranked job matches with an **explainable score**, matched and missing skills, and a list of skills that would unlock the most roles. Built with NLP, embeddings and a hybrid ranking model, served through a FastAPI backend and a web UI.

## How it works

```
resume ─► text extraction ─► sections ─► skill extraction ─► role inference + experience estimate
                                                   │
 job sources ─► normalise ─► requirement parsing ──┤
 (Remotive, Arbeitnow, RemoteOK, Adzuna)           ▼
                                      hybrid ranker (5 features)
                                                   ▼
                       ranked jobs + score breakdown + skill-gap report
```

| Stage | What it does |
|---|---|
| Parsing | `pdfplumber` for PDFs, XML parsing for DOCX, section detection (skills, experience, education, projects) |
| Skill extraction | 200+ term taxonomy with aliases (`k8s` → Kubernetes, `sklearn` → scikit-learn); skills listed in a Skills section get a weight boost |
| Role inference | IDF-style scoring of 13 role profiles (skills shared by fewer roles count more) |
| Experience | Union of employment date ranges (education lines ignored), mapped to a level |
| Embeddings | Pluggable backend: `all-MiniLM-L6-v2` sentence-transformers when installed, otherwise TF-IDF + truncated SVD (LSA) |
| Hybrid ranker | `semantic` + `lexical` (TF-IDF cosine) + `skills` (IDF-weighted coverage, role-defining skills count more than generic ones, partial credit for related skills) + `role fit` + `level fit` |
| Explainability | Per-feature scores, matched skills, transferable skills (e.g. needs TensorFlow, you know PyTorch), missing required skills |
| Skill-gap report | Which missing skills appear in the most shortlisted jobs |

## Quick start

```bash
pip install -r requirements.txt
python -m jobmatch my_resume.pdf --location Chennai          # CLI, offline demo jobs
python -m jobmatch my_resume.pdf --live                      # real listings from job APIs
uvicorn jobmatch.api:app --reload                            # web UI at http://localhost:8000
pytest                                                       # 15 tests
```

Optional: `pip install sentence-transformers` for dense embeddings (auto-detected). For India coverage set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` (free keys at developer.adzuna.com).

Docker: `docker build -t jobmatch . && docker run -p 8000:8000 jobmatch`

### API

`POST /api/match` (multipart): `file` or `text`, `location`, `remote_only`, `live`, `top_k` → profile, ranked matches with score components, `skills_to_learn`, per-source status.

## Evaluation (read the caveat)

`python scripts/evaluate.py` runs 130 held-out synthetic resumes against 78 synthetic jobs across 13 role families. Relevance: same role family = 2, adjacent family = 1, else 0.

| Strategy | P@5 | MRR | nDCG@10 |
|---|---|---|---|
| TF-IDF only | 0.954 | 0.983 | 0.940 |
| Semantic only (LSA) | 0.903 | 0.971 | 0.911 |
| Skill coverage only | 0.974 | 0.996 | 0.950 |
| **Hybrid (default weights)** | **0.980** | **0.992** | **0.953** |
| Hybrid (learned weights, logistic regression) | 0.985 | 0.992 | 0.953 |

**Caveat:** resumes and jobs are generated from the same skill taxonomy, so absolute numbers are optimistic. This is a sanity check and a *relative* comparison of strategies, not a claim about real-world accuracy. Real validation needs human-labelled resume–job pairs.

## Limitations

- Skill extraction is dictionary based; skills outside the taxonomy are not detected (extend `jobmatch/skills.py`).
- Free job APIs mostly list remote and international roles. LinkedIn, Naukri and Indeed have no open search APIs and scraping them breaks their terms, so they are not scraped.
- The bundled sample jobs are synthetic and flagged `source: sample`.
- Years of experience come from date ranges and can be wrong for unusual formats.
- Only English resumes are supported.

## Roadmap

- Fine-tune a cross-encoder on labelled resume–job pairs
- Section-aware embeddings (match projects to responsibilities)
- LLM-generated tailored cover-letter bullets per match
- Saved searches and email alerts

## Project layout

```
jobmatch/   skills.py parser.py embeddings.py ranker.py connectors.py matcher.py api.py __main__.py
static/     web UI
scripts/    make_sample_jobs.py  evaluate.py
tests/      15 tests (parsing, ranking, mocked connectors, API)
```
