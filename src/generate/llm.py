"""
Ollama LLM Provider with Structured Pydantic Output Enforcement
===============================================================
Per RepoTriage_SPEC.md Section 7 and Stage 4 requirements:
- Uses local Ollama at http://localhost:11434 with qwen3.5:4b.
- Uses standard-library urllib only (no third-party HTTP dependencies).
- Strictly enforces Pydantic JSON schema format.
- Disables thinking mode (think: False) for speed and deterministic JSON.
- 1-retry fallback on invalid JSON or schema parse error.
- Post-validates evidence IDs against supplied items (E1..En).
- Provides unload_model() with keep_alive: 0 for strict VRAM isolation.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from src.generate.prompts import SYSTEM_PROMPT, build_analysis_prompt
from src.generate.schema import AnalysisResult, Claim

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_NUM_CTX = 8192
FALLBACK_NUM_CTX = 6144
DEFAULT_NUM_PREDICT = 1200
DEFAULT_TEMPERATURE = 0.2


class OllamaClient:
    """Standard-library urllib client for local Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_MODEL,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_NUM_PREDICT,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def unload_model(self) -> bool:
        """Free model from Ollama GPU VRAM by sending keep_alive: 0."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [],
            "keep_alive": 0,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                success = (resp.status == 200)
                if success:
                    logger.info("Successfully unloaded %s from Ollama VRAM", self.model_name)
                return success
        except Exception as e:
            logger.warning("Could not unload Ollama model %s: %s", self.model_name, e)
            return False

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send chat request to Ollama and return parsed JSON response."""
        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logger.error("Ollama HTTP Error %d: %s", e.code, err_body)
            raise RuntimeError(f"Ollama HTTP error {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            logger.error("Failed to connect to Ollama at %s: %s", self.base_url, e)
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Is Ollama running? (Error: {e})"
            ) from e

    def generate_analysis(
        self,
        issue_text: str,
        repo: str | None = None,
        evidence_items: Sequence[dict[str, Any]] | None = None,
        predicted_type: str | None = None,
        auto_unload: bool = False,
    ) -> tuple[AnalysisResult, dict[str, Any]]:
        """
        Generate structured issue analysis and claims list via Ollama.

        Parameters
        ----------
        issue_text : str
            Title and body of incoming issue.
        repo : str | None
            Repository name, e.g. 'pydantic/pydantic'.
        evidence_items : Sequence[dict] | None
            Top 5-6 retrieved evidence items with evidence_id ('E1'..'En').
        predicted_type : str | None
            Output from frozen classifier.
        auto_unload : bool
            Whether to automatically free Ollama VRAM after generation.

        Returns
        -------
        tuple[AnalysisResult, dict]
            Validated Pydantic AnalysisResult and metadata execution trace.
        """
        evidence_items = list(evidence_items or [])
        valid_evidence_ids = {
            item.get("evidence_id") or f"E{idx}"
            for idx, item in enumerate(evidence_items, 1)
        }

        user_prompt = build_analysis_prompt(
            issue_text=issue_text,
            repo=repo,
            evidence_items=evidence_items,
            predicted_type=predicted_type,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "format": AnalysisResult.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        t0 = time.perf_counter()
        retried = False
        raw_response: dict[str, Any] = {}
        content_str = ""

        try:
            raw_response = self._post_chat(payload)
            content_str = raw_response.get("message", {}).get("content", "").strip()
            parsed_json = json.loads(content_str)
            result = AnalysisResult.model_validate(parsed_json)
        except Exception as first_exc:
            logger.warning("First generation attempt failed: %s. Retrying once...", first_exc)
            retried = True
            retry_messages = list(messages)
            if content_str:
                retry_messages.append({"role": "assistant", "content": content_str})
            retry_messages.append({
                "role": "user",
                "content": f"The previous output failed validation: {first_exc}. "
                           "Please return ONLY a valid JSON object strictly conforming to the schema.",
            })
            payload["messages"] = retry_messages
            # Fall back to 6144 ctx if 8192 encountered an issue
            if self.num_ctx > FALLBACK_NUM_CTX:
                payload["options"]["num_ctx"] = FALLBACK_NUM_CTX

            raw_response = self._post_chat(payload)
            content_str = raw_response.get("message", {}).get("content", "").strip()
            try:
                parsed_json = json.loads(content_str)
                result = AnalysisResult.model_validate(parsed_json)
            except Exception as second_exc:
                raise RuntimeError(
                    f"Ollama generation failed after 1 retry: {second_exc}\nRaw output: {content_str[:500]}"
                ) from second_exc

        gen_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Post-process claims: discard invalid or invented evidence IDs
        sanitized_claims: list[Claim] = []
        for c in result.claims:
            cleaned_ids = [eid for eid in c.evidence_ids if eid in valid_evidence_ids]
            sanitized_claims.append(
                Claim(
                    claim_id=c.claim_id,
                    text=c.text.strip(),
                    evidence_ids=cleaned_ids,
                )
            )
        result.claims = sanitized_claims

        trace = {
            "model": self.model_name,
            "retried": retried,
            "generation_latency_ms": round(gen_latency_ms, 2),
            "prompt_eval_count": raw_response.get("prompt_eval_count", 0),
            "eval_count": raw_response.get("eval_count", 0),
            "num_claims": len(result.claims),
            "num_cited_claims": sum(1 for c in result.claims if c.evidence_ids),
        }

        if auto_unload:
            self.unload_model()

        return result, trace
