from __future__ import annotations

from pathlib import Path

from database import delete_user_data, get_messages, list_scans, save_messages, save_scan
from chatbot import OllamaError, build_system_prompt, stream_chat
from knowledge import retrieve
from report import build_pdf_report
from triage import assess_triage
from PIL import Image


def test_red_flag_triage_is_urgent() -> None:
    result = assess_triage({"breathing_difficulty": "Yes"}, "Normal")
    assert result.level == "Red"
    assert "breathing difficulty" in result.reasons


def test_non_red_flags_with_suspected_class_prompt_review() -> None:
    result = assess_triage({"fever": "No"}, "Measles")
    assert result.level == "Yellow"
    assert "24–48 hours" in result.action


def test_incomplete_or_uncertain_safety_screen_is_not_green() -> None:
    incomplete = assess_triage({}, "Normal")
    uncertain = assess_triage(
        {
            "age": "34",
            "pregnancy": "Not sure / prefer not to say",
            "very_high_fever": "No",
            "immunocompromised": "No",
            "breathing_difficulty": "No",
            "seizures": "No",
            "confusion": "No",
            "rash_near_eyes": "No",
        },
        "Normal",
    )
    assert incomplete.level == "Yellow"
    assert uncertain.level == "Yellow"


def test_complete_negative_screen_and_normal_model_can_be_green() -> None:
    result = assess_triage(
        {
            "age": "34",
            "pregnancy": "No",
            "very_high_fever": "No",
            "immunocompromised": "No",
            "breathing_difficulty": "No",
            "seizures": "No",
            "confusion": "No",
            "rash_near_eyes": "No",
        },
        "Normal",
    )
    assert result.level == "Green"


def test_local_history_requires_explicit_function_call_and_deletes(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite3"
    scan_id = save_scan("profile-1", "Uncertain", 0.42, "Yellow", "Fever: No", db_path=db_path)
    save_messages(scan_id, [{"role": "user", "content": "What now?"}], db_path=db_path)
    save_messages(scan_id, [
        {"role": "user", "content": "What now?"},
        {"role": "assistant", "content": "Please seek care."},
    ], db_path=db_path)
    assert len(list_scans("profile-1", db_path=db_path)) == 1
    assert [message["role"] for message in get_messages(scan_id, db_path=db_path)] == ["user", "assistant"]
    assert delete_user_data("profile-1", db_path=db_path) == 1
    assert list_scans("profile-1", db_path=db_path) == []
    assert get_messages(scan_id, db_path=db_path) == []


def test_local_sources_retrievable() -> None:
    results = retrieve("measles rash cough fever", limit=2)
    assert results
    assert any("CDC" in result.source for result in results)


def test_prompt_carries_model_context_and_refuses_missing_model() -> None:
    triage = assess_triage({"pregnancy": "Yes"}, "Chickenpox")
    prompt = build_system_prompt("Chickenpox", 0.72, {"pregnancy": "Yes"}, triage)
    assert "Chickenpox" in prompt
    assert "Red" in prompt
    try:
        list(stream_chat("", [], prompt))
    except OllamaError:
        pass
    else:
        raise AssertionError("An empty model name must produce an explicit Ollama error")


def test_pdf_report_is_generated_with_a_disclaimer() -> None:
    image = Image.new("RGB", (40, 40), "teal")
    pdf = build_pdf_report(
        "Uncertain", 0.51, {"Chickenpox": 0.51, "Measles": 0.2, "Monkeypox": 0.19, "Normal": 0.1},
        "Yellow", "Contact a healthcare professional.", "Fever: <script>alert(1)</script>",
        "General advice only.", image, image,
    )
    assert pdf.startswith(b"%PDF")
