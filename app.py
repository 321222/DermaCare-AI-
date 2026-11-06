"""DermaCare AI: local-first skin-health support application."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import streamlit as st
from PIL import Image, ImageStat

from audio_utils import synthesize_speech, transcribe_audio
from chatbot import (
    DISCLAIMER,
    OllamaError,
    build_system_prompt,
    list_models,
    response_sources,
    stream_chat,
)
from database import (
    delete_user_data,
    get_messages,
    list_scans,
    save_messages,
    save_scan,
    update_scan_summary,
)
from report import build_pdf_report
from triage import TriageResult, assess_triage, emergency_banner

CLASS_NAMES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = Path(os.environ.get("DERMACARE_MODEL_PATH", MODEL_DIR / "skin_disease_cnn.keras"))
LABELS_PATH = Path(os.environ.get("DERMACARE_LABELS_PATH", MODEL_DIR / "class_names.json"))
CONFIDENCE_THRESHOLD = 0.60
DISCLAIMER_SHORT = (
    "AI-assisted skin-health information—not a diagnosis or a substitute for professional care. "
    "Image results require clinician assessment; laboratory testing may be needed."
)
INTAKE_QUESTIONS: list[tuple[str, str, str]] = [
    ("age", "How old is the person with the rash?", "text"),
    ("sex", "What sex was recorded at birth? (optional)", "text"),
    ("pregnancy", "Is the person pregnant or could they be pregnant?", "yesno"),
    ("duration", "How long has the rash been present?", "text"),
    ("fever", "Is there a fever?", "yesno"),
    ("very_high_fever", "Is the fever very high, persistent, or getting worse?", "yesno"),
    ("itching", "Is the rash itchy?", "yesno"),
    ("cough", "Is there a cough?", "yesno"),
    ("red_eyes", "Are the eyes red or watery?", "yesno"),
    ("swollen_glands", "Are there swollen glands?", "yesno"),
    ("travel_exposure", "Any recent travel or close contact with someone ill?", "text"),
    ("mmr_vaccination", "Is MMR vaccination up to date, if known?", "text"),
    ("varicella_vaccination", "Is varicella (chickenpox) vaccination up to date, if known?", "text"),
    ("conditions", "Any existing health conditions?", "text"),
    ("immunocompromised", "Is the person immunocompromised or taking immune-suppressing treatment?", "yesno"),
    ("allergies", "Any allergies?", "text"),
    ("breathing_difficulty", "Any difficulty breathing?", "yesno"),
    ("seizures", "Any seizure?", "yesno"),
    ("confusion", "Any confusion or unusual drowsiness?", "yesno"),
    ("rash_near_eyes", "Is the rash on or near the eyes?", "yesno"),
]


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str) -> Any:
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def image_quality_notes(image: Image.Image) -> list[str]:
    notes: list[str] = []
    if min(image.size) < 224:
        notes.append("The image is small; fine details may not be visible.")
    gray = image.convert("L")
    mean_luminance = ImageStat.Stat(gray).mean[0]
    if mean_luminance < 35:
        notes.append("The image appears very dark; consider a brighter, evenly lit photo.")
    elif mean_luminance > 225:
        notes.append("The image appears overexposed; consider reducing glare.")
    if ImageStat.Stat(gray).var[0] < 35:
        notes.append("The image may be blurry or have low contrast.")
    return notes


def run_inference(image: Image.Image, model: Any) -> tuple[str, float, dict[str, float], np.ndarray]:
    resized = image.convert("RGB").resize((224, 224))
    base = np.asarray(resized, dtype=np.float32)
    variants = [
        base,
        np.flip(base, axis=1).copy(),
        np.clip((base - 127.5) * 1.05 + 127.5, 0, 255),
    ]
    predictions = model.predict(np.stack(variants), verbose=0)
    probabilities = np.mean(predictions, axis=0)
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8")) if LABELS_PATH.exists() else CLASS_NAMES
    if labels != CLASS_NAMES:
        raise ValueError(f"Model class labels do not match expected classes: {labels}")
    result = CLASS_NAMES[index] if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"
    return result, confidence, dict(zip(CLASS_NAMES, map(float, probabilities))), base


def summarize_intake(intake: dict[str, Any]) -> str:
    return "; ".join(f"{key.replace('_', ' ').capitalize()}: {value}"
                     for key, value in intake.items() if str(value).strip())


def normalize_red_flags(intake: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(intake)
    age = str(intake.get("age", "")).lower()
    match = re.search(r"\d+", age)
    if "newborn" in age:
        normalized["newborn"] = True
    elif match:
        value = int(match.group())
        normalized["newborn"] = (
            ("day" in age and value <= 28)
            or ("week" in age and value <= 4)
            or ("month" in age and value == 0)
        )
    conditions = str(intake.get("conditions", "")).lower()
    immune_answer = intake.get("immunocompromised")
    immune_condition = any(
        word in conditions
        for word in ("immune suppressed", "immunocompromised", "transplant", "chemotherapy")
    )
    if immune_answer is not None:
        answer = str(immune_answer).strip().lower()
        if answer in {"yes", "no", "true", "false"}:
            normalized["immunocompromised"] = answer in {"yes", "true"} or immune_condition
        else:
            normalized["immunocompromised"] = immune_answer
    elif immune_condition:
        normalized["immunocompromised"] = True
    return normalized


def current_prediction() -> str | None:
    data = st.session_state.get("scan_result")
    return data.get("prediction") if data else None


def current_triage() -> TriageResult:
    return assess_triage(normalize_red_flags(st.session_state.get("intake", {})), current_prediction())


def on_quick_reply(value: str) -> None:
    st.session_state["pending_prompt"] = value


def save_current_session_if_consented() -> None:
    result = st.session_state.get("scan_result")
    if (
        not result
        or not st.session_state.get("storage_consent")
        or st.session_state.get("storage_suppressed_after_delete")
    ):
        return
    old_hash = st.session_state.get("persisted_image_hash")
    if (
        old_hash == result["image_hash"]
        and st.session_state.get("persisted_user_id") == st.session_state["user_id"]
    ):
        scan_id = st.session_state.get("persisted_scan_id")
        if scan_id is not None:
            triage = current_triage()
            update_scan_summary(
                int(scan_id), triage.level, summarize_intake(st.session_state.get("intake", {}))
            )
        return
    triage = current_triage()
    scan_id = save_scan(
        st.session_state["user_id"], result["prediction"], result["confidence"],
        triage.level, summarize_intake(st.session_state.get("intake", {})),
        result["image_hash"],
    )
    st.session_state["persisted_image_hash"] = result["image_hash"]
    st.session_state["saved_scan_id"] = scan_id
    st.session_state["persisted_scan_id"] = scan_id
    st.session_state["persisted_user_id"] = st.session_state["user_id"]


def _show_intake() -> None:
    st.subheader("Symptom and safety check")
    intake = st.session_state.setdefault("intake", {})
    unanswered = next((item for item in INTAKE_QUESTIONS if item[0] not in intake), None)
    completed = sum(1 for key, _, _ in INTAKE_QUESTIONS if key in intake)
    st.progress(completed / len(INTAKE_QUESTIONS), text=f"{completed} of {len(INTAKE_QUESTIONS)} questions answered")
    if unanswered is None:
        st.success("Check complete. You can review or restart the questions.")
        if st.button("Start intake again", key="reset_intake"):
            st.session_state["intake"] = {}
            for key, _, _ in INTAKE_QUESTIONS:
                st.session_state.pop(f"answer_{key}", None)
            st.rerun()
        return
    key, question, kind = unanswered
    with st.form("intake_form"):
        st.write(f"**{question}**")
        if kind == "yesno":
            answer = st.radio("Choose one", ["Yes", "No", "Not sure / prefer not to say"],
                              horizontal=True, key=f"answer_{key}")
        else:
            answer = st.text_input("Your answer (or leave blank)", key=f"answer_{key}")
        submitted = st.form_submit_button("Save and continue")
    if submitted:
        intake[key] = answer.strip() if isinstance(answer, str) else answer
        st.rerun()
    if st.button("Skip this question", key=f"skip_{key}"):
        intake[key] = "Not provided"
        st.rerun()


def _detection_tab() -> None:
    st.markdown("Upload a well-lit, focused photo of the affected area. The image is processed on this device and is not sent to Ollama.")
    uploaded = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png"],
        key="skin_photo",
        help="JPG or PNG, up to 10 MB. Avoid including faces or identifying details.",
    )
    if uploaded is None:
        st.info(f"A trained model file (`{MODEL_PATH}`) is required for image classification. See the project README for training instructions.")
        return
    try:
        image = Image.open(uploaded)
        image.verify()
        uploaded.seek(0)
        image = Image.open(uploaded).convert("RGB")
    except (OSError, ValueError) as exc:
        st.error(f"Could not read the selected image: {exc}")
        return
    image_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
    st.image(image, caption="Uploaded image (processed locally)", use_container_width=True)
    notes = image_quality_notes(image)
    for note in notes:
        st.warning(note)
    if st.button("Analyze image", type="primary"):
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(MODEL_PATH)
            model = load_classifier(str(MODEL_PATH))
            prediction, confidence, probabilities, _ = run_inference(image, model)
            gradcam_image = None
            try:
                from gradcam import make_gradcam_heatmap, overlay_heatmap

                batch = np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0)
                raw_class = max(probabilities, key=probabilities.get)
                heatmap = make_gradcam_heatmap(batch, model, CLASS_NAMES.index(raw_class))
                gradcam_image = overlay_heatmap(image, heatmap)
            except (ImportError, ValueError, RuntimeError) as exc:
                st.info(f"Grad-CAM is unavailable for this model: {exc}")
            st.session_state["scan_result"] = {
                "prediction": prediction, "confidence": confidence, "probabilities": probabilities,
                "image": image.copy(), "gradcam": gradcam_image, "image_hash": image_hash,
                "quality_notes": notes,
            }
            save_current_session_if_consented()
        except FileNotFoundError:
            st.error(f"Trained model not found at `{MODEL_PATH}`. Run `python train.py` after preparing the dataset.")
        except ImportError:
            st.error("TensorFlow is not installed. Install the dependencies from requirements.txt.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Image analysis failed: {exc}")

    result = st.session_state.get("scan_result")
    if not result or result["image_hash"] != image_hash:
        return
    if result["prediction"] == "Uncertain":
        st.warning(f"**Uncertain result:** the top class confidence is {result['confidence']:.1%}, below the {CONFIDENCE_THRESHOLD:.0%} threshold. Do not rely on this image result.")
    else:
        st.markdown(
            f"<div class='result-card'><b>Model suggestion: {result['prediction']}</b>"
            f"<br>Confidence: {result['confidence']:.1%} · Not a diagnosis</div>",
            unsafe_allow_html=True,
        )
    st.subheader("Class probabilities")
    colors_by_class = {
        "Chickenpox": "#ea580c",
        "Measles": "#dc2626",
        "Monkeypox": "#7c3aed",
        "Normal": "#059669",
    }
    probability_columns = st.columns(2)
    for index, (label, probability) in enumerate(result["probabilities"].items()):
        color = colors_by_class[label]
        probability_columns[index % 2].markdown(
            f"<div class='probability-card' style='--class-color:{color}'>"
            f"<b>{label}</b><span>{probability:.1%}</span>"
            f"<div class='probability-track'><i style='width:{probability * 100:.2f}%'></i></div></div>",
            unsafe_allow_html=True,
        )
    if result["gradcam"] is not None:
        st.image(result["gradcam"], caption="Grad-CAM (coarse model attention visualization; not clinical evidence)", width=400)
    triage = current_triage()
    if triage.level == "Red":
        st.error(emergency_banner(triage))
    elif triage.level == "Yellow":
        st.warning(f"**Yellow — prompt medical review.** {triage.action}")
    else:
        st.success(f"**Green — no urgent warning signs reported.** {triage.action}")
    save_current_session_if_consented()


def _chat_tab() -> None:
    st.markdown(f"> {DISCLAIMER}")
    triage = current_triage()
    if current_prediction() or st.session_state.get("intake"):
        st.caption(f"Safety check: {triage.level} · {triage.action}")
    else:
        st.caption("Complete the symptom and safety check or add a photo before requesting a care summary.")
    quick_cols = st.columns(4)
    for column, prompt in zip(quick_cols, ["How does it spread?", "Home care", "When to see a doctor?", "Prevention"]):
        column.button(prompt, key=f"quick_{prompt}", on_click=on_quick_reply, args=(prompt,))
    if st.session_state.get("voice_enabled"):
        audio_input = st.audio_input("Record a question")
        if audio_input is not None:
            audio_hash = hashlib.sha256(audio_input.getvalue()).hexdigest()
            if st.session_state.get("transcribed_hash") != audio_hash:
                try:
                    transcript = transcribe_audio(audio_input.getvalue())
                    st.session_state["pending_prompt"] = transcript
                    st.session_state["transcribed_hash"] = audio_hash
                    st.success(f"Transcribed locally: {transcript}")
                except (RuntimeError, ValueError) as exc:
                    st.warning(str(exc))

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.get("messages"):
        st.info(
            "Hello, I’m Dr. Derma—an AI assistant, not a human doctor. "
            "Tell me what’s concerning you. Nothing is saved unless you opt in to local history."
        )
    typed_prompt = st.chat_input("Ask Dr. Derma about symptoms or care", key="chat_message")
    if typed_prompt:
        st.session_state.pop("pending_prompt", None)
    prompt = typed_prompt or st.session_state.pop("pending_prompt", None)
    if prompt and current_triage().level == "Red":
        urgent_response = (
            "I’m concerned by the urgent warning sign you reported. Please seek urgent medical care now. "
            "If symptoms are severe or worsening, contact local emergency services. I can’t assess or "
            "manage an emergency through chat."
        )
        st.session_state["messages"].extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": urgent_response},
        ])
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(urgent_response)
        save_current_session_if_consented()
        scan_id = st.session_state.get("saved_scan_id")
        if scan_id and st.session_state.get("storage_consent"):
            save_messages(scan_id, st.session_state["messages"])
        prompt = None
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        triage = current_triage()
        sources = response_sources(f"{prompt} {current_prediction() or ''}")
        source_text = "\n".join(f"- {source.source}" for source in sources)
        source_excerpts = sources
        language = "the language used by the user (English, Hindi, or Marathi)"
        system_prompt = build_system_prompt(
            current_prediction(),
            st.session_state.get("scan_result", {}).get("confidence"),
            normalize_red_flags(st.session_state.get("intake", {})),
            triage,
            language,
            source_excerpts,
        )
        with st.chat_message("assistant"):
            answer_area = st.empty()
            accumulated = ""
            try:
                for token in stream_chat(
                    st.session_state.get("selected_model", ""),
                    st.session_state["messages"],
                    system_prompt,
                ):
                    accumulated += token
                    answer_area.markdown(accumulated + "▌")
                if not accumulated.strip():
                    raise OllamaError("The model returned an empty reply.")
                answer_area.markdown(accumulated)
                if source_text:
                    st.caption(f"Local reference excerpts: {source_text.replace(chr(10), ', ')}")
                    for source in sources:
                        reference = re.search(r"Reference:\s*(https?://\S+)", source.text)
                        if reference:
                            st.markdown(f"Source: [{source.source}]({reference.group(1)})")
                if st.session_state.get("voice_enabled"):
                    try:
                        spoken, mime = synthesize_speech(
                            accumulated,
                            allow_online_fallback=st.session_state.get("allow_online_tts", False),
                            language=st.session_state.get("tts_language", "en"),
                        )
                        st.audio(spoken, format=mime)
                    except (RuntimeError, ValueError) as exc:
                        st.info(str(exc))
                st.session_state["messages"].append({"role": "assistant", "content": accumulated})
                save_current_session_if_consented()
                scan_id = st.session_state.get("saved_scan_id")
                if scan_id and st.session_state.get("storage_consent"):
                    save_messages(scan_id, st.session_state["messages"])
            except OllamaError as exc:
                answer_area.error(str(exc))
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "I couldn't connect to the local language model. Please check Ollama and try again.",
                })


def _history_tab() -> None:
    st.subheader("Your local history")
    st.caption(
        "History is stored unencrypted on this device and only after explicit consent. "
        "This local profile code is a lookup key—not a login, identity check, or access control. "
        "Do not use this feature on a shared or publicly accessible computer."
    )
    profile = st.text_input("Local profile code", key="user_id", help="Keep this code private; anyone with access to the device and code may view the saved history.")
    scans = list_scans(profile) if profile.strip() else []
    if not scans:
        st.info("No saved scans for this profile code.")
    for scan in scans:
        with st.expander(f"{scan['created_at'][:16]} · {scan['prediction']} · {scan['triage_level']}"):
            st.write(f"Model confidence: {scan['confidence']:.1%}")
            st.write(scan["symptom_summary"] or "No symptoms recorded")
            for message in get_messages(int(scan["id"])):
                st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")
    st.divider()
    confirm_delete = st.checkbox("I understand this permanently deletes all saved scans and chats for this profile.")
    if st.button("Delete my saved data", disabled=not confirm_delete or not profile.strip()):
        removed = delete_user_data(profile)
        st.session_state["storage_suppressed_after_delete"] = True
        st.session_state["saved_scan_id"] = None
        st.session_state["persisted_scan_id"] = None
        st.session_state["persisted_image_hash"] = None
        st.session_state["persisted_user_id"] = None
        st.session_state["scan_result"] = None
        st.session_state["messages"] = []
        st.session_state["intake"] = {}
        st.success(f"Deleted {removed} saved scan(s) and their conversations.")
        st.rerun()


def _reports_tab() -> None:
    st.subheader("Consultation summary")
    result = st.session_state.get("scan_result")
    if not result:
        st.info("Analyze an image in Detection to prepare a report.")
        return
    triage = current_triage()
    summary = summarize_intake(st.session_state.get("intake", {}))
    advice = (
        f"{triage.action} Use gentle, non-irritating skin care and avoid sharing personal items. "
        "Ask a healthcare professional about isolation, prevention, vaccination, and medicines. "
        "Do not give aspirin to children with possible chickenpox; confirm any medicine with a doctor or pharmacist."
    )
    try:
        pdf = build_pdf_report(
            result["prediction"], result["confidence"], result["probabilities"],
            triage.level, triage.action, summary, advice, result["image"], result["gradcam"],
        )
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not build the PDF report: {exc}")
        return
    st.download_button(
        "Download PDF report", data=pdf, file_name="dermacare-consultation-summary.pdf",
        mime="application/pdf", type="primary",
    )
    st.caption("The PDF includes the AI disclaimer, model limitations, image, model probabilities, triage and general next steps.")


def _about_tab() -> None:
    st.subheader("About DermaCare AI")
    st.markdown(
        """
        **Dr. Derma is an AI assistant, not a human clinician.** DermaCare AI classifies
        images into Chickenpox, Measles, Monkeypox (mpox), or Normal. Skin conditions can look alike;
        this model cannot diagnose and cannot rule out illness. A clinician may need an examination and
        laboratory confirmation, such as PCR.

        **Dataset limitations:** results may be unreliable across skin-tone diversity, lighting, camera
        quality, body site, lesion stage, and image framing. Model metrics are not evidence of
        clinical safety, generalization, or fitness for medical decision-making. Never use this tool to delay care.

        **Local setup:** put images under `dataset/Chickenpox`, `dataset/Measles`, `dataset/Monkeypox`,
        and `dataset/Normal`; run `python train.py`, then `streamlit run app.py`. Start Ollama and pull
        a model such as `ollama pull llama3.2`. See `README.md` for detailed setup and limits.
        """
    )
    st.subheader("Emergency warning signs")
    st.write("Breathing difficulty, seizure, confusion, very high or persistent fever, rash near the eyes, newborn age, pregnancy, or immune suppression need urgent medical advice. Contact emergency services for severe or rapidly worsening symptoms.")
    st.link_button(
        "Find a doctor near me",
        "https://www.google.com/maps/search/" + quote_plus("doctor near me"),
    )
    st.caption("Google Maps opens an external service. No location is accessed by this app.")


def main() -> None:
    st.set_page_config(page_title="DermaCare AI", page_icon="🩺", layout="wide")
    st.markdown(
        """
        <style>
        :root { color-scheme: light dark; }
        .hero { padding: 1.65rem 2rem; border-radius: 18px; margin: .2rem 0 1.2rem;
                background: linear-gradient(118deg, #075e68 0%, #087f8c 52%, #2563a6 100%);
                color: #fff; box-shadow: 0 10px 28px rgba(8, 80, 100, .16); }
        .hero h1 { margin: 0; color: #fff; letter-spacing: -.035em; font-size: 2.2rem; }
        .hero p { margin: .35rem 0 0; color: rgba(255,255,255,.9); font-size: 1rem; }
        .hero .eyebrow { margin-bottom: .35rem; font-size: .75rem; font-weight: 700;
                         letter-spacing: .14em; text-transform: uppercase; }
        .result-card { padding: 1.1rem 1.2rem; border: 1px solid rgba(8,127,140,.4);
                       border-left: 5px solid #087f8c; border-radius: 12px; margin: .8rem 0; }
        .probability-card { padding: .9rem 1rem; border: 1px solid rgba(127,127,127,.16);
                            border-top: 4px solid var(--class-color); border-radius: 12px;
                            background: rgba(127, 127, 127, .06); margin: .35rem 0; }
        .probability-card span { float: right; font-variant-numeric: tabular-nums; font-weight: 650; }
        .probability-track { height: 7px; border-radius: 6px; background: rgba(127, 127, 127, .2);
                             margin-top: .7rem; overflow: hidden; }
        .probability-track i { display: block; height: 100%; background: var(--class-color);
                               border-radius: inherit; }
        [data-testid="stTabs"] button { font-weight: 600; }
        [data-testid="stFileUploader"] { border-radius: 14px; }
        @media (max-width: 700px) {
            .hero { padding: 1.1rem; border-radius: 14px; }
            .hero h1 { font-size: 1.7rem; }
        }
        </style>
        <div class="hero"><div class="eyebrow">Private skin-health support</div>
        <h1>DermaCare AI</h1><p>Clear next steps, with Dr. Derma</p></div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(DISCLAIMER_SHORT, icon="⚕️")
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("intake", {})
    with st.sidebar:
        st.header("Preferences")
        st.caption("Image analysis stays on this device. Dr. Derma chat uses your local Ollama service.")
        st.checkbox(
            "Save my scans and chat history on this device",
            key="storage_consent",
            help="Optional. History is stored in unencrypted local SQLite. Unchecked means new history is not saved.",
        )
        if not st.session_state.get("storage_consent"):
            st.session_state["storage_suppressed_after_delete"] = False
        if st.session_state.get("storage_suppressed_after_delete"):
            st.caption("After deleting data, saving stays disabled until you turn consent off and opt in again.")
        try:
            models = list_models()
            if models:
                current = st.session_state.get("selected_model")
                index = models.index(current) if current in models else 0
                st.session_state["selected_model"] = st.selectbox("Ollama model", models, index=index)
                st.caption("Ollama is available.")
            else:
                st.session_state["selected_model"] = ""
                st.warning("Ollama is running, but no models were found. Pull a model such as `llama3.2`.")
        except OllamaError as exc:
            st.session_state["selected_model"] = ""
            st.warning(str(exc))
        st.checkbox("Enable voice input/output", key="voice_enabled", value=False)
        if st.session_state.get("voice_enabled"):
            st.selectbox("Speech reply language", ["en", "hi", "mr"], key="tts_language")
            st.checkbox(
                "Allow online gTTS fallback if offline speech is unavailable",
                key="allow_online_tts",
                value=False,
                help="gTTS sends reply text to Google's service. Leave unchecked for local-only processing.",
            )
        if st.button("Clear this chat from memory"):
            st.session_state["messages"] = []
            st.rerun()

    with st.expander("Symptom and safety check", expanded=False):
        _show_intake()
    banner = emergency_banner(current_triage())
    if banner:
        st.error(banner)
    tabs = st.tabs(["Detection", "Dr. Derma Chat", "History", "Reports", "About"])
    with tabs[0]:
        _detection_tab()
    with tabs[1]:
        _chat_tab()
    with tabs[2]:
        _history_tab()
    with tabs[3]:
        _reports_tab()
    with tabs[4]:
        _about_tab()


if __name__ == "__main__":
    main()
