"""
Pydantic Schemas for Structured Issue Analysis and Fact Claims
=============================================================
Per RepoTriage_SPEC.md Section 7:
Output schema strictly enforced via Ollama's JSON schema mode.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """An atomic factual claim made in the analysis, grounded in retrieved evidence."""

    claim_id: str = Field(
        description="Unique sequential identifier for this claim, e.g. 'C1', 'C2', 'C3'",
    )
    text: str = Field(
        description="Single atomic factual statement about repository behavior, documentation, errors, or solutions",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="List of cited evidence IDs supporting this claim, e.g. ['E1', 'E2']. Empty list if ungrounded.",
    )


class AnalysisResult(BaseModel):
    """Full structured triage and resolution analysis for a GitHub issue."""

    issue_summary: str = Field(
        description="Concise 1-2 sentence summary of the core defect, question, or feature request",
    )
    issue_type: Literal["bug", "feature", "question", "uncertain"] = Field(
        description="Classified category of the issue report",
    )
    affected_components: list[str] = Field(
        default_factory=list,
        description="Names of affected modules, classes, functions, or API endpoints",
    )
    observed_symptoms: list[str] = Field(
        default_factory=list,
        description="Observed exceptions, error messages, tracebacks, or incorrect return values",
    )
    environment_details: str = Field(
        default="not established",
        description="Python version, OS, dependency versions, or 'not established' if omitted",
    )
    reproduction_steps: list[str] = Field(
        default_factory=list,
        description="Minimal steps or reproducible code snippet to trigger the behavior",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Key information missing from the issue report needed for diagnosis",
    )
    likely_causes: list[str] = Field(
        default_factory=list,
        description="Technical root causes or contributing factors suggested by code/evidence",
    )
    recommended_diagnostics: list[str] = Field(
        default_factory=list,
        description="Debugging steps, log commands, or inspection points to confirm root cause",
    )
    recommended_next_steps: list[str] = Field(
        default_factory=list,
        description="Concrete resolution actions, workarounds, code edits, or PR references",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Extracted atomic factual claims with citations to supplied evidence (E1..En)",
    )
    limitations: str = Field(
        default="Local triage synthesis; external dependencies or recent unindexed changes may differ.",
        description="Caveats or boundaries of this automated local diagnosis",
    )
