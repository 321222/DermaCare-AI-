🩺 DermaCare AI

A CNN-based skin-disease screening app (Chickenpox, Measles, Monkeypox, Normal) with a voice-enabled, local Ollama chatbot that explains results, suggests care, and covers prevention and impact.

Show Image Show Image Show Image Show Image

⚠️ Medical disclaimer: This project is for education and screening support only. It is not a medical diagnosis and does not replace a doctor. These diseases look alike in photos; always confirm with a qualified clinician and lab tests.

✨ Features
Image classification: a custom CNN classifies a skin photo as Chickenpox, Measles, Monkeypox or Normal, with a confidence score and per-class probabilities.
Low-confidence warning: results under 60% confidence are flagged as unreliable.
Health information: suggestions (including when to see a doctor), prevention and impact for each condition.
AI assistant: a local Ollama chatbot that uses the prediction as context and streams its replies.
Voice: ask questions by microphone (offline Whisper speech-to-text) and hear results and replies (text-to-speech in English, Hindi, Marathi).
Professional UI: Streamlit interface with detection, assistant and about tabs.
Privacy-friendly: the CNN and the LLM run on your own machine.
Evaluation outputs: accuracy/loss curves, confusion matrix, and precision/recall/F1 report.
🏗️ How it works
 Skin image ──► CNN (TensorFlow/Keras) ──► class + confidence
                                              │
                                              ▼
 Voice / text question ──► Whisper (STT) ──► Ollama LLM (Llama 3.2 / Qwen3)
                                              │      ▲ prediction + curated facts
                                              ▼      │
                                      Answer (stream) ──► gTTS / pyttsx3 (TTS)
📁 Project structure
skin_disease_app/
├── train.py           # CNN training and evaluation
├── app.py             # Streamlit UI (detection, voice, chatbot)
├── knowledge.py       # Suggestions / prevention / impact content
├── chatbot.py         # Ollama streaming client + system prompt
├── audio_utils.py     # Text-to-speech and speech-to-text helpers
├── requirements.txt
├── dataset/           # (you add this) one folder per class
├── skin_disease_cnn.keras   # created by train.py
├── class_names.json         # created by train.py
├── training_curves.png      # created by train.py
└── confusion_matrix.png     # created by train.py
🚀 Getting started
1. Prerequisites
Python 3.10, 3.11 or 3.12
Ollama installed
A microphone (optional, for voice questions)
2. Install
bash
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
3. Get the dataset

Download a skin-lesion image dataset with these four classes (for example, the Monkeypox Skin Lesion Dataset on Kaggle) and arrange it like this. Check the dataset's licence and terms before using it.

dataset/
├── Chickenpox/  *.jpg
├── Measles/     *.jpg
├── Monkeypox/   *.jpg
└── Normal/      *.jpg
4. Train the CNN
bash
python train.py --data_dir dataset --epochs 30 --batch_size 32

This creates skin_disease_cnn.keras, class_names.json, training_curves.png and confusion_matrix.png, and prints a classification report.

💻 On a laptop without a GPU? Train on Google Colab (free T4 GPU), then download skin_disease_cnn.keras and class_names.json into this folder. You can also lower IMG_SIZE (e.g. 128) in both train.py and app.py.

5. Start Ollama and pull a model
bash
ollama serve
ollama pull llama3.2:3b     # fast, good default for laptops
ollama pull qwen3:4b        # optional: stronger reasoning / multilingual, slower
ollama list                 # confirm installed models
6. Run the app
bash
streamlit run app.py

Open the URL shown in the terminal (usually http://localhost:8501).

🧭 Using the app
Detection tab: upload a clear, well-lit photo or use the camera. You get the prediction, confidence, probability bars, and Suggestions / Prevention / Impact tabs. Click Listen to summary to hear it.
AI Assistant tab: type a question, click a quick question, or use the microphone. The assistant already knows the detection result.
Sidebar: pick the Ollama model, set reply creativity, choose the speech language (English, Hindi, Marathi), toggle spoken replies, and choose the speech-recognition size (tiny / base / small).
⚙️ Configuration
Setting	Where	Default
Image size	IMG_SIZE in train.py and app.py	224
Low-confidence threshold	CONFIDENCE_THRESHOLD in app.py	0.60
Ollama host	Sidebar	http://localhost:11434
Fallback model name	app.py	llama3.2:3b
Whisper model	Sidebar	base
📊 Results

Fill these in after training:

Metric	Value
Test accuracy	TBD
Macro F1-score	TBD
Training images	TBD

Add confusion_matrix.png and training_curves.png here once generated.

🛠️ Troubleshooting
Problem	Fix
"Model files not found"	Run python train.py first.
Sidebar says Ollama is offline	Run ollama serve, check the host in the sidebar.
Chatbot is slow	Use llama3.2:3b, close other apps, avoid training at the same time.
st.audio_input error	Upgrade Streamlit: pip install -U "streamlit>=1.40".
Voice question fails	Install faster-whisper; the first use downloads the model.
No spoken audio	Install gTTS (needs internet) or pyttsx3 (offline).
TensorFlow install issues on Windows	Use a supported Python version, or WSL2.
⚠️ Limitations
Trained on a limited public dataset; lighting, skin tone, camera quality and body site can change results.
Measles, chickenpox and mpox can look similar; confirmation needs a clinician and laboratory testing.
The LLM can make mistakes. The prompt tells it not to diagnose or give drug doses, but its answers still need checking.
The health text in knowledge.py is a general summary. Verify it against WHO or CDC before wider use.
🔮 Future work
Transfer learning (MobileNetV2 / EfficientNet) for higher accuracy
Grad-CAM heatmaps to show what the model looked at
Symptom intake and triage levels inside the chatbot
Retrieval-augmented answers from WHO/CDC documents
PDF report download and a patient-history dashboard
🧰 Tech stack

Python · TensorFlow/Keras · scikit-learn · Streamlit · Ollama (Llama 3.2, Qwen3) · faster-whisper · gTTS · pyttsx3 · NumPy · Matplotlib · Seaborn
