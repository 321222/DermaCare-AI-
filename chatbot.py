"""Local Ollama chat integration with safety rules and source grounding."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from knowledge import SourceExcerpt
from rag import get_relevant_context
from triage import TriageResult

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
REQUEST_TIMEOUT = 90
DISCLAIMER = (
    "I’m Dr. Derma, an AI assistant—not a human doctor. I can’t diagnose from a photo. "
    "A clinician may need an examination and lab testing (such as PCR) to confirm a cause."
)


class OllamaError(RuntimeError):
    """A user-displayable Ollama connectivity or model error."""


def list_models(base_url: str = OLLAMA_URL, timeout: int = 5) -> list[str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise OllamaError("Ollama is unavailable. Start Ollama locally and try again.") from exc
    return sorted(
        str(model.get("name", "")).strip()
        for model in payload.get("models", [])
        if model.get("name")
    )


def build_system_prompt(
    prediction: str | None,
    confidence: float | None,
    intake: Mapping[str, Any],
    triage: TriageResult,
    language: str = "the language used by the user",
    sources: Sequence[SourceExcerpt] = (),
) -> str:
    source_text = "\n\n".join(f"[{item.source}]\n{item.text}" for item in sources) or "No matching local source."
    confidence_text = f"{confidence:.1%}" if confidence is not None else "not available"
    return f"""You are Dr. Derma, a careful, empathetic AI skin-health assistant.
{DISCLAIMER}

SAFETY RULES:
- Never claim a diagnosis or that a photo proves a disease. Say "the model suggests" and recommend clinician confirmation; laboratory confirmation such as PCR may be needed.
- Give general information only. Never provide medication doses or prescribe. Advise checking any medicine with a doctor or pharmacist. Never recommend aspirin for a child with possible chickenpox.
- If any red flag is present, repeat urgent/emergency care advice clearly and early; do not delay care with questions.
- Do not infer missing facts. Ask only one concise intake question at a time when more information is needed.
- When relevant, cover comfort-focused home care, spread/prevention, possible complications, follow-up, and isolation/contact precautions. Isolation duration depends on the suspected illness and local public-health advice; do not invent a fixed duration.
- If asked for an interpretation or care plan, briefly organize it as: model suggestion and uncertainty, comfort-focused care, spread/prevention and locally guided precautions, possible impacts/complications, next steps/follow-up, and warning signs.
- Refuse unrelated or unsafe requests politely and redirect to skin-health support.
- Be empathetic, brief, accessible, and non-alarmist. Do not overstate the accuracy of this model or sources.
- Reply in {language}; preserve the user's language (English, Hindi, or Marathi).
- If asked about the image result, explain that different conditions can look alike and briefly discuss plausible alternatives without pretending to distinguish them from this image. State that clinician assessment and possibly laboratory testing such as PCR are needed for confirmation.
- Cite only the local source labels provided below, e.g. [CDC]. Do not invent references.

CURRENT SESSION (user-provided / model-generated, not verified):
Image model suggestion: {prediction or "not available"}
Model confidence: {confidence_text}
Intake answers: {json.dumps(dict(intake), ensure_ascii=False, default=str)}
Rule-based triage: {triage.level}; reasons: {", ".join(triage.reasons)}; action: {triage.action}

LOCAL REFERENCE EXCERPTS:
{source_text}
"""


def stream_chat(
    model: str,
    messages: Sequence[Mapping[str, str]],
    system_prompt: str,
    base_url: str = OLLAMA_URL,
    timeout: int = REQUEST_TIMEOUT,
) -> Iterator[str]:
    """Yield text chunks from Ollama's newline-delimited /api/chat response."""
    if not model.strip():
        raise OllamaError("Choose an installed Ollama model first.")
    body = json.dumps({
        "model": model,
        "stream": True,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "options": {"temperature": 0.3},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                if chunk.get("error"):
                    raise OllamaError(str(chunk["error"]))
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield str(text)
                if chunk.get("done"):
                    return
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 404:
            raise OllamaError(f"Model '{model}' is missing. Pull it with `ollama pull {model}`.") from exc
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise OllamaError("Could not reach Ollama. Check that it is running and try again.") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned an unreadable response.") from exc


def response_sources(user_message: str) -> list[SourceExcerpt]:
    return get_relevant_context(user_message)
