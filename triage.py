"""Rule-based safety triage for the DermaCare AI application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TRIAGE_LEVELS = ("Green", "Yellow", "Red")


@dataclass(frozen=True)
class TriageResult:
    level: str
    reasons: tuple[str, ...]
    action: str


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"yes", "true", "1", "present"}


def _uncertain(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "not sure",
        "not sure / prefer not to say",
        "unknown",
        "unsure",
    }


def _answered(value: Any) -> bool:
    return bool(str(value or "").strip()) and str(value).strip().lower() not in {
        "not provided",
        "prefer not to say",
    }


def assess_triage(intake: Mapping[str, Any], prediction: str | None = None) -> TriageResult:
    """Return a cautious rule-based care-seeking level, not a diagnosis."""
    emergency_flags = {
        "breathing_difficulty": "breathing difficulty",
        "seizures": "seizures",
        "confusion": "confusion or unusual drowsiness",
        "very_high_fever": "very high or persistent fever",
        "rash_near_eyes": "rash near the eyes",
        "newborn": "a newborn with a rash",
        "pregnancy": "pregnancy with a new rash",
        "immunocompromised": "a weakened immune system",
    }
    red_reasons = [label for key, label in emergency_flags.items() if _truthy(intake.get(key))]
    if red_reasons:
        return TriageResult(
            "Red",
            tuple(red_reasons),
            "Seek urgent medical care now. If symptoms are severe or worsening, contact local emergency services.",
        )

    required_screening = (
        "age",
        "pregnancy",
        "very_high_fever",
        "immunocompromised",
        "breathing_difficulty",
        "seizures",
        "confusion",
        "rash_near_eyes",
    )
    screening_incomplete = any(not _answered(intake.get(key)) for key in required_screening)

    uncertain_red_flags = [
        key.replace("_", " ")
        for key in emergency_flags
        if _uncertain(intake.get(key))
    ]
    concerning = any(
        _truthy(intake.get(key))
        for key in ("fever", "swollen_glands", "rash_worsening", "eye_pain")
    ) or any(_uncertain(intake.get(key)) for key in ("pregnancy", "very_high_fever"))
    uncertain_model = prediction in (None, "Uncertain", "Error")
    suspected_infection = prediction in {"Chickenpox", "Measles", "Monkeypox"}
    if concerning or uncertain_model or suspected_infection or screening_incomplete:
        reason = "The image result is not a diagnosis and a new or worsening rash deserves clinical assessment."
        if concerning:
            reason = "Reported symptoms make prompt clinical assessment prudent."
        elif screening_incomplete:
            reason = "The safety check is incomplete, so urgent warning signs have not been assessed."
        if uncertain_red_flags:
            reason += " Some urgent warning signs are uncertain: " + ", ".join(uncertain_red_flags) + "."
        return TriageResult(
            "Yellow",
            (reason,),
            (
                "Complete the safety check and contact a healthcare professional within 24–48 hours, "
                "or sooner if symptoms worsen."
                if screening_incomplete and not suspected_infection and not concerning
                else "Contact a healthcare professional within 24–48 hours, or sooner if symptoms worsen."
            ),
        )

    return TriageResult(
        "Green",
        ("No urgent warning signs were reported in the information provided.",),
        "This check cannot rule out a medical problem. Use gentle comfort measures and seek routine medical advice if the rash persists, changes, or worries you.",
    )


def emergency_banner(result: TriageResult) -> str | None:
    """Format an explicit emergency message for prominent UI display."""
    if result.level != "Red":
        return None
    details = ", ".join(result.reasons)
    return f"URGENT: {details.capitalize()}. {result.action}"
