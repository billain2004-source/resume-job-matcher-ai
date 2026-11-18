"""Skill taxonomy, alias normalisation, role profiles and skill extraction."""
from __future__ import annotations
import math
import re
from collections import Counter

# canonical skill -> aliases (all lowercase)
TAXONOMY: dict[str, list[str]] = {
    # languages
    "python": ["python3"], "java": [], "javascript": ["js", "ecmascript", "es6"],
    "typescript": ["ts"], "c++": ["cpp"], "c#": ["csharp", "c sharp"], "golang": ["go lang"],
    "rust": [], "php": [], "ruby": [], "kotlin": [], "swift": [], "scala": [], "dart": [],
    "r programming": ["r language"], "sql": ["t-sql", "pl/sql"], "bash": ["shell scripting", "shell script"],
    # web
    "html": ["html5"], "css": ["css3"], "react": ["react.js", "reactjs"], "angular": ["angularjs"],
    "vue": ["vue.js", "vuejs"], "next.js": ["nextjs"], "gatsby": ["gatsby.js", "gatsbyjs"],
    "node.js": ["nodejs"], "express": ["express.js", "expressjs"], "redux": [],
    "tailwind css": ["tailwind", "tailwindcss"], "bootstrap": [], "graphql": [],
    "rest api": ["rest apis", "restful api", "restful apis", "restful"], "wordpress": [],
    "webpack": [],
    # backend
    "django": [], "flask": [], "fastapi": [], "spring boot": ["springboot"], "spring": [],
    ".net": ["dotnet", "asp.net"], "microservices": ["microservice"], "kafka": ["apache kafka"],
    "rabbitmq": [],
    # data stores
    "mysql": [], "postgresql": ["postgres"], "mongodb": ["mongo"], "redis": [], "sqlite": [],
    "elasticsearch": ["elastic search"], "nosql": [], "snowflake": [],
    # cloud / devops
    "aws": ["amazon web services", "ec2", "s3"], "azure": ["microsoft azure"],
    "gcp": ["google cloud", "google cloud platform"], "docker": ["containerization"],
    "kubernetes": ["k8s"], "terraform": [], "ansible": [], "jenkins": [],
    "github actions": [], "ci/cd": ["cicd", "ci cd", "continuous integration"], "linux": ["ubuntu"],
    "git": [], "github": [], "gitlab": [], "nginx": [], "prometheus": [], "grafana": [],
    # ML / AI
    "machine learning": ["ml"], "deep learning": [], "nlp": ["natural language processing"],
    "computer vision": [], "tensorflow": [], "pytorch": [], "keras": [],
    "scikit-learn": ["sklearn", "scikit learn"], "pandas": [], "numpy": [], "opencv": [],
    "transformers": ["hugging face", "huggingface"], "llm": ["llms", "large language models", "large language model"],
    "langchain": [], "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "mlops": [], "mlflow": [], "xgboost": [], "spark": ["pyspark", "apache spark"], "hadoop": [],
    "airflow": ["apache airflow"], "feature engineering": [], "statistics": ["statistical analysis"],
    "a/b testing": ["ab testing", "a/b tests"],
    # analytics
    "power bi": ["powerbi"], "tableau": [], "excel": ["microsoft excel", "ms excel"],
    "data analysis": ["data analytics"], "data visualization": ["data visualisation"],
    "etl": [], "looker": [], "google analytics": [],
    # mobile
    "android": [], "ios": [], "flutter": [], "react native": [], "swiftui": [], "firebase": [],
    # QA
    "selenium": [], "cypress": [], "playwright": [], "pytest": [], "junit": [],
    "manual testing": [], "test automation": ["automation testing", "automated testing"],
    "postman": [], "jmeter": [],
    # security
    "cybersecurity": ["cyber security", "information security", "infosec"],
    "penetration testing": ["pentesting", "pen testing", "ethical hacking"], "owasp": [],
    "siem": [], "splunk": [], "wireshark": [], "nmap": [], "burp suite": [],
    "network security": [], "vulnerability assessment": ["vulnerability management"],
    # customer / marketing / business
    "customer support": ["technical support", "customer service"],
    "customer success": [], "crm": [], "hubspot": [], "salesforce": [], "zendesk": [],
    "freshdesk": [], "email marketing": [], "email deliverability": ["deliverability"],
    "seo": ["search engine optimization", "search engine optimisation"],
    "lead generation": ["lead gen"], "fraud detection": ["fraud analysis", "fraud review"],
    "jira": [], "agile": [], "scrum": [], "figma": [], "content writing": ["copywriting"],
    "google ads": ["google adwords"],
    # CS fundamentals
    "data structures": ["dsa"], "algorithms": [], "oop": ["object oriented programming", "object-oriented programming"],
    "system design": [], "design patterns": [], "dbms": ["database management"],
    "operating systems": [], "computer networks": ["computer networking"],
}

ROLE_PROFILES: dict[str, list[str]] = {
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "pytorch", "tensorflow",
                                  "scikit-learn", "numpy", "pandas", "mlops", "docker", "nlp",
                                  "computer vision", "llm", "transformers", "mlflow", "aws"],
    "Data Scientist": ["python", "statistics", "machine learning", "pandas", "numpy", "scikit-learn", "sql",
                       "data visualization", "a/b testing", "feature engineering", "xgboost", "spark"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "python", "data analysis", "data visualization",
                     "etl", "statistics", "google analytics", "looker"],
    "Frontend Developer": ["javascript", "typescript", "react", "html", "css", "redux", "next.js",
                           "tailwind css", "rest api", "git", "angular", "vue", "webpack", "gatsby"],
    "Backend Developer": ["python", "java", "node.js", "django", "flask", "fastapi", "spring boot",
                          "postgresql", "mongodb", "redis", "rest api", "microservices", "docker", "kafka", "sql"],
    "Full Stack Developer": ["javascript", "react", "node.js", "express", "mongodb", "postgresql", "rest api",
                             "html", "css", "typescript", "docker", "git"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "terraform", "ansible", "jenkins", "ci/cd", "linux",
                        "github actions", "prometheus", "grafana", "bash", "azure", "gcp"],
    "Mobile App Developer": ["flutter", "react native", "android", "kotlin", "ios", "swift", "firebase",
                             "rest api", "dart", "swiftui", "git"],
    "QA Automation Engineer": ["selenium", "cypress", "playwright", "pytest", "test automation",
                               "manual testing", "postman", "jmeter", "java", "ci/cd", "junit", "jira"],
    "Cybersecurity Analyst": ["cybersecurity", "siem", "splunk", "owasp", "penetration testing", "wireshark",
                              "nmap", "network security", "vulnerability assessment", "linux", "burp suite", "python"],
    "Customer Success Specialist": ["customer success", "customer support", "crm", "hubspot", "salesforce",
                                    "zendesk", "freshdesk", "email deliverability", "jira", "sql", "agile"],
    "WordPress Developer": ["wordpress", "php", "html", "css", "javascript", "mysql", "gatsby", "seo",
                            "react", "git", "bootstrap"],
    "Digital Marketing Specialist": ["seo", "email marketing", "google ads", "lead generation",
                                     "google analytics", "hubspot", "content writing", "email deliverability",
                                     "excel", "crm"],
}

# roles close enough that a match on one is partially relevant for the other
ADJACENT_ROLES: dict[str, set[str]] = {
    "Machine Learning Engineer": {"Data Scientist"},
    "Data Scientist": {"Machine Learning Engineer", "Data Analyst"},
    "Data Analyst": {"Data Scientist"},
    "Frontend Developer": {"Full Stack Developer", "WordPress Developer"},
    "Backend Developer": {"Full Stack Developer", "DevOps Engineer"},
    "Full Stack Developer": {"Frontend Developer", "Backend Developer"},
    "DevOps Engineer": {"Backend Developer"},
    "WordPress Developer": {"Frontend Developer", "Digital Marketing Specialist"},
    "Customer Success Specialist": {"Digital Marketing Specialist"},
    "Digital Marketing Specialist": {"Customer Success Specialist", "WordPress Developer"},
}

# partial credit groups: a missing skill is "transferable" when you know a sibling
RELATED_GROUPS: list[set[str]] = [
    {"mysql", "postgresql", "sqlite", "sql", "snowflake"},
    {"react", "angular", "vue", "next.js", "gatsby"},
    {"aws", "azure", "gcp"},
    {"pytorch", "tensorflow", "keras"},
    {"tableau", "power bi", "looker"},
    {"selenium", "cypress", "playwright"},
    {"flask", "django", "fastapi"},
    {"jenkins", "github actions", "gitlab", "ci/cd"},
    {"hubspot", "salesforce", "crm"},
    {"zendesk", "freshdesk", "customer support"},
    {"kotlin", "android", "java"},
    {"flutter", "react native"},
    {"docker", "kubernetes"},
    {"mongodb", "nosql", "redis"},
    {"machine learning", "deep learning", "scikit-learn"},
]

_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in TAXONOMY.items():
    _ALIAS_TO_CANON[_canon] = _canon
    for _a in _aliases:
        _ALIAS_TO_CANON[_a] = _canon

_NOT_FRAMEWORK = re.compile(r"\s+(?:interest|concern|gratitude|consent|delivery|shipping|yourself|your|their|my|our|how|that|the|a)\b", re.I)
_pattern_cache: re.Pattern | None = None


def _build_pattern() -> re.Pattern:
    global _pattern_cache
    if _pattern_cache is None:
        names = sorted(_ALIAS_TO_CANON, key=len, reverse=True)
        alt = "|".join(re.escape(n) for n in names)
        _pattern_cache = re.compile(rf"(?<![a-z0-9+#.])({alt})(?![a-z0-9+#])", re.IGNORECASE)
    return _pattern_cache


def normalize_skill(name: str) -> str | None:
    return _ALIAS_TO_CANON.get(name.strip().lower())


def find_skills(text: str) -> Counter:
    """Count canonical skills mentioned in text, resolving aliases."""
    counts: Counter = Counter()
    has_node = bool(re.search(r"node\.?js|\bnode\b", text, re.I))
    for m in _build_pattern().finditer(text):
        raw = m.group(1).lower()
        if raw == "express" and (not has_node or _NOT_FRAMEWORK.match(text, m.end())):
            continue                              # "express interest" is not the framework
        counts[_ALIAS_TO_CANON[raw]] += 1
    return counts


def extract_skills(text: str, skills_section: str = "") -> dict[str, float]:
    """Weighted skills: mentions (log-damped) plus a boost for appearing in a skills section."""
    found = find_skills(text)
    boosted = find_skills(skills_section) if skills_section else Counter()
    weights: dict[str, float] = {}
    for s, c in found.items():
        weights[s] = 1.0 + math.log1p(c - 1) * 0.5 + (0.5 if s in boosted else 0.0)
    return weights


def infer_roles(skills: dict[str, float] | set[str] | list[str], top: int = 4) -> list[tuple[str, float]]:
    """Score every role profile; skills shared by fewer roles count more (like IDF)."""
    present = set(skills)
    n_roles = Counter(s for prof in ROLE_PROFILES.values() for s in set(prof))
    scores: dict[str, float] = {}
    for role, prof in ROLE_PROFILES.items():
        hit = sum(1.0 / n_roles[s] for s in set(prof) if s in present)
        scores[role] = hit / math.sqrt(len(prof))
    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(r, s / total) for r, s in ranked[:top] if s > 0]


def related(skill: str, have: set[str]) -> str | None:
    for grp in RELATED_GROUPS:
        if skill in grp:
            for h in have:
                if h in grp and h != skill:
                    return h
    return None
