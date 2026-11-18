from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class Job:
    id: str
    title: str
    company: str
    description: str
    location: str = ""
    remote: bool = False
    url: str = ""
    source: str = "sample"
    required: list[str] = field(default_factory=list)
    preferred: list[str] = field(default_factory=list)
    min_years: float | None = None
    role_label: str = ""          # ground-truth family, only set on the synthetic corpus

    def text(self) -> str:
        return f"{self.title}. {self.description} {' '.join(self.required)}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Profile:
    text: str
    skills: dict[str, float]            # canonical skill -> weight (frequency / section boost)
    roles: list[tuple[str, float]]      # inferred roles with confidence, best first
    years: float                        # estimated years of experience
    level: str                          # intern | entry | junior | mid | senior
    sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skills": sorted(self.skills, key=self.skills.get, reverse=True),
            "roles": [{"role": r, "confidence": round(c, 3)} for r, c in self.roles],
            "years": round(self.years, 1),
            "level": self.level,
        }


@dataclass
class Match:
    job: Job
    score: float
    components: dict[str, float]
    matched: list[str]
    transferable: list[tuple[str, str]]   # (job skill, resume skill that partly covers it)
    missing_required: list[str]
    missing_preferred: list[str]

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict(),
            "score": round(self.score * 100, 1),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "matched": self.matched,
            "transferable": [{"needs": a, "you_have": b} for a, b in self.transferable],
            "missing_required": self.missing_required,
            "missing_preferred": self.missing_preferred,
        }
