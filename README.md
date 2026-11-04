DermaCare AI

DermaCare AI is a local-first skin-health support application combining a four-class image classifier with an Ollama-powered conversational assistant, “Dr. Derma.” It provides a professional interface and a foundation for controlled evaluation; it is **not clinically validated, does not diagnose, and is not a substitute for clinical care**. Images can be misleading; a clinician may need an examination and laboratory confirmation such as PCR.

## What it includes

- Streamlit UI: Detection, Dr. Derma Chat, History, Reports, and About.
- CNN training: stratified 70/15/15 image split, augmentation, class weights, custom four-block CNN vs. MobileNetV2 with fine-tuning, callbacks, and held-out evaluation reports.
- Inference: horizontal-flip/contrast test-time augmentation, 60% uncertainty threshold, quality notices, per-class probabilities, and Grad-CAM visualization.
- Ollama: local model discovery through `/api/tags`, selectable model, streamed `/api/chat`, dynamic safety/context prompt, and graceful offline/model-missing messages.
- Local reference excerpts from the included WHO/CDC/NHS educational summaries.
- Structured intake, rule-based Green/Yellow/Red safety triage, voice options, downloadable PDF, and consent-gated SQLite history with delete-my-data.
- Responsive light/dark-aware interface, clear medical disclaimer, and dataset limitation notice.

## Project map

```text
Browser
  └── app.py (Streamlit UI and in-memory session state)
       ├── train.py / skin_disease_cnn.keras ── image model
       ├── triage.py ─────────────────────────── red-flag rules
       ├── chatbot.py ── localhost:11434 ─────── Ollama /api/chat stream
       │    └── knowledge.py / rag.py ───────── local WHO/CDC/NHS text
       ├── audio_utils.py ────────────────────── offline speech tools
       ├── database.py ── local SQLite ───────── opt-in history
       └── report.py ─────────────────────────── in-memory PDF
```

No uploaded image is sent to Ollama. History is off unless the user opts in; when enabled, it is stored in a local SQLite file (`dermacare_history.sqlite3`). Ollama receives chat text and the current model prediction/intake context. Optional gTTS sends reply text to Google's service and is disabled unless explicitly enabled; prefer the offline `pyttsx3` option. Local processing does not by itself guarantee secure deployment or compliant handling of health data.

## Setup

Use Python 3.10–3.12. TensorFlow and faster-whisper can download large dependencies/models. First create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install/start [Ollama](https://ollama.com/) locally, then pull a model:

```powershell
ollama pull llama3.2
ollama serve
```

Ollama defaults to `http://localhost:11434`. To use another local endpoint, set `OLLAMA_HOST`. The app lists installed model names in the sidebar and gives an actionable message if the service or selected model is unavailable.

## Prepare data and train

Use an authorized, appropriately labeled dataset. Arrange it as:

```text
dataset/
  Chickenpox/*.jpg
  Measles/*.jpg
  Monkeypox/*.jpg
  Normal/*.jpg
```

The trainer also accepts `.jpeg` and `.png`. It requires at least seven images in each class to create a stratified split. Start training from this directory:

```powershell
python train.py --data dataset --output models --epochs 25 --fine-tune-epochs 10
```

The selected model and `class_names.json` are written under `models/`, which is ignored by Git because trained model artifacts should not be published by default. Training outputs include a terminal/CSV model comparison table and JSON summary, accuracy/loss curves, confusion matrices, classification reports, and one-vs-rest ROC curves. `evaluate.py` evaluates the saved model against a labeled dataset directory:

```powershell
python evaluate.py --model models/skin_disease_cnn.keras --dataset dataset --output evaluation
```

**Evaluation limitation:** the requested split is stratified by image, not by patient. Duplicate or same-patient images can leak across splits. Do not treat these metrics as clinical validation. A real medical evaluation requires patient-level splits, representative skin tones/body sites and external data, expert adjudication, prospective validation, and appropriate governance.

## Run

```powershell
streamlit run app.py
```

Upload a JPG/PNG and select **Analyze image**. Complete the intake one question at a time to refine rule-based triage and conversation context. The app loads model artifacts from `models/` by default (or paths configured by `DERMACARE_MODEL_PATH` and `DERMACARE_LABELS_PATH`), and marks predictions under 60% as **Uncertain**. Grad-CAM is only a coarse view of model attention; it is not evidence that a lesion is present or a clinical explanation.

To keep history, opt in using the consent checkbox in the sidebar. A local profile code is shown in History; it is a convenience key, **not authentication**. Use **Delete my saved data** to remove records for that code. Without consent, the app does not persist scans/chat. To use the camera microphone, enable voice in the sidebar; faster-whisper processes speech locally after its model is available on disk (the first use may download model files). Speech output prefers offline pyttsx3. gTTS requires a separate explicit opt-in because it transmits text externally.

## Docker Compose: Streamlit + Ollama + Cloudflare Tunnel

The Compose stack runs Streamlit and Ollama on a private Docker network. Ollama's API is **not published to the host or tunnel**. Streamlit binds to `127.0.0.1:8501`; Cloudflare Tunnel is optional and is the only service that can provide remote access.

1. Install Docker Desktop and create the local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Start the private app stack, then download one of the model weights into the Compose Ollama volume:

   ```powershell
   docker compose up -d --build
   docker compose exec ollama ollama pull llama3.2:3b
   # Or: docker compose exec ollama ollama pull qwen3:4b
   ```

   The tags already installed in Windows Ollama are separate from Docker's Ollama model volume. Pulling into the container is a one-time download; persistent data stays in the `ollama_data` Docker volume.

3. Open `http://localhost:8501`. For image detection, put the trained `skin_disease_cnn.keras` and `class_names.json` files in `models/`. `models/` is deliberately excluded from Git and the Docker build context.

4. To enable Cloudflare, create a Cloudflare Tunnel in the Zero Trust dashboard and configure a public hostname service URL of `http://streamlit:8501`. Before starting the tunnel, protect that hostname with a **Cloudflare Access application** and restrict it to intended users (for example, an email one-time-pin policy). Put the tunnel token in `.env` as `CLOUDFLARE_TUNNEL_TOKEN=...`, then:

   ```powershell
   docker compose --profile public up -d --build
   ```

   Never commit `.env` or expose the Ollama API. Do not make the hostname publicly reachable without Access: this app has no user authentication, and its local profile code is not an access-control mechanism. A tunnel encrypts transport but does not by itself authenticate users or make local health data secure. Review Cloudflare logs, retention, privacy, and applicable health-data requirements before remote use.

5. View service status/logs or stop the stack:

   ```powershell
   docker compose ps
   docker compose logs -f streamlit ollama
   docker compose down
   ```

   `docker compose down` preserves named volumes. To remove stored history/model downloads, use Docker's volume removal only after confirming the data should be permanently deleted.

## GitHub

The repository includes `.github/workflows/ci.yml`, which runs the focused tests on pushes to `main`, pull requests, and manual dispatch. The workflow does not publish an image or deploy a tunnel; those operations require an explicitly configured repository, registry policy, and Cloudflare secret. Add the project to a GitHub repository you control, verify that no `.env`, dataset, trained weights, SQLite history, or patient data are staged, then push the branch to trigger CI.

To upload the GitHub-ready source package manually, download `DermaCare-AI-GitHub-Source.zip`, extract it, and upload the contents of its `DermaCare-AI-/` folder to your repository using GitHub's web upload feature. For the cleaner, repeatable Git path, install Git for Windows and authenticate through Git Credential Manager or GitHub Desktop. Then run the included publisher from PowerShell:

```powershell
.\publish-to-github.ps1 -DryRun
.\publish-to-github.ps1
```

The script clones `321222/DermaCare-AI-` into a temporary folder, preserves the existing repository license, copies only the application/source/docs/CI files and empty data/model folders, reviews the staged summary, and requires you to type `PUSH` before committing and pushing to `main`. It refuses to stage model weights, datasets, images, database files, or secret files. The dry run makes no remote changes. If Git/GitHub authentication is unavailable, install Git for Windows or GitHub Desktop and complete browser sign-in first.

## Product and clinical readiness

The software is structured as a runnable application, but the included image-model workflow and local history are not certified or production-authorized for clinical use. Clinical deployment would require, at minimum, representative and permissioned data, patient-level external validation, calibration and subgroup analysis, prospective study, clinician oversight, formal risk management, privacy/security controls, and jurisdiction-specific regulatory review. Do not remove the clinical limitations because the interface looks professional.

## Medical safety

- The interface and every PDF carry a disclaimer. The model suggests a class; it never confirms or excludes a condition.
- Red flags include breathing difficulty, seizures, confusion, very high/persistent fever, rash near the eyes, a newborn, pregnancy, and immune suppression. Red triage advises urgent care; call local emergency services for severe or rapidly worsening symptoms.
- Medication guidance is general only and does not include doses. Confirm medicines with a doctor or pharmacist. Avoid aspirin in children with possible chickenpox.
- For suspected measles/mpox or another contagious illness, contact a healthcare provider/public-health service before attending in person; follow local isolation and contact advice.
- Dataset and image limitations include skin-tone diversity, lighting, focus/camera quality, body site, lesion stage, and class representation.
- The bundled text excerpts are short, curated educational summaries with source URLs—not a continuously updated guideline database. Verify current advice with the linked authorities and local public-health services.

## Screenshots

Capture screenshots after launching the app and add them here when available:

| Screen | Screenshot |
|---|---|
| Detection and class probabilities | _Add `screenshots/detection.png`_ |
| Dr. Derma consultation and triage | _Add `screenshots/consultation.png`_ |
| History and PDF report | _Add `screenshots/history-report.png`_ |

## Testing checklist

- [ ] Training completes and reports accuracy, per-class precision/recall/F1, confusion matrix, ROC, and model comparison.
- [ ] Ollama offline produces a clear error; installed models appear; missing model gives `ollama pull ...` guidance.
- [ ] Low-confidence prediction below 60% is shown as **Uncertain**.
- [ ] Each red-flag answer displays urgent-care advice.
- [ ] Voice transcription works locally; TTS works offline or explains how to enable online fallback.
- [ ] PDF downloads with image/Grad-CAM (when available), symptom summary, triage, advice, and disclaimer.
- [ ] History is not saved before consent, is scoped to the local profile code, and can be deleted.
- [ ] Check the interface in light/dark mode and a narrow/mobile viewport.

Run the focused non-model tests:

```powershell
pytest -q
```

## Extending the project

1. Replace or augment the dataset with licensed, clinically reviewed examples and add patient-level metadata for leakage-resistant splits.
2. Benchmark calibration and subgroup performance; consider abstention/review workflows instead of making a classifier more decisive.
3. Keep triage rules reviewed by qualified clinicians and localize emergency guidance, vaccine schedules, and isolation advice to jurisdiction.
4. Update the cited local knowledge files with dated, source-verified text; for larger RAG collections, add local embeddings only after evaluating retrieval quality.
5. Add authentication, encryption-at-rest, retention controls, audit logging, and a privacy/security review before any shared or clinical deployment.

## Screenshots of training artifacts

Training plots and reports are generated locally and ignored by Git; see the commands above to create them. No sample medical predictions or fabricated performance metrics are included in this repository.
