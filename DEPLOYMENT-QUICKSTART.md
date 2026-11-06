# DermaCare AI: GitHub and Cloudflare deployment kit

This folder is the source bundle for the GitHub repository `321222/DermaCare-AI-`.
It does not include trained model weights, training images, local SQLite history,
or Cloudflare secrets. Keep it that way when publishing.

## 1. Publish the source to GitHub

1. Extract `DermaCare-AI-GitHub-Source.zip` to a normal folder, for example
   `C:\Users\<you>\Desktop\DermaCare-AI-Deploy`.
2. Install Git for Windows or GitHub Desktop and sign in to GitHub in the app.
3. Open PowerShell in the extracted folder. For example:

   ```powershell
   Set-Location "$HOME\Desktop\DermaCare-AI-Deploy\DermaCare-AI-"
   ```

4. Preview the changes first:

   ```powershell
   .\publish-to-github.ps1 -DryRun
   ```

5. If the staged-file summary looks correct, run:

   ```powershell
   .\publish-to-github.ps1
   ```

6. Type `PUSH` only after reviewing the summary. The script clones the existing
   repository, preserves its current license, stages a fixed allow-list, and
   refuses to publish model files, images, databases, or secrets.

If GitHub Desktop is already installed, its bundled Git should be detected by the
script. If PowerShell says the script is not recognized, first `Set-Location` to
the extracted project folder. Do not run it from `C:\Windows\System32`.

## 2. Add model artifacts locally

The GitHub repository intentionally does not contain model weights or datasets.
For detection, train with your authorized dataset or supply a model, then place:

```text
models/
  skin_disease_cnn.keras
  class_names.json
```

Never put patient images or personal health data in this deployment folder or
public repository.

## 3. Run Streamlit and Ollama with Docker

Install Docker Desktop, open a terminal in the project folder, and run:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b
```

The Ollama service runs inside Docker and has a persistent model volume. Models
installed in Windows Ollama are separate; pull the model into the Compose service.
Open `http://localhost:8501`. Put the trained files under `models/` before using
image classification.

## 4. Enable Cloudflare Tunnel

Create a Cloudflare Tunnel in the Cloudflare dashboard and add a public hostname
route with service `http://streamlit:8501`. Protect the hostname with a
Cloudflare Access application restricted to intended people **before** exposing
it. Add the tunnel token to the local `.env` file:

```dotenv
CLOUDFLARE_TUNNEL_TOKEN=your-token
```

Then start the public profile:

```powershell
docker compose --profile public up -d
```

Do not commit `.env`. Tunnel encryption does not replace login/access controls,
and this application does not implement user authentication. The local profile
code is not an account. Do not expose health history publicly.

## Services and verification

```powershell
docker compose ps
docker compose logs -f streamlit ollama cloudflared
docker compose down
```

GitHub Actions runs the test suite when changes are pushed. A successful CI run
does not validate the skin classifier for clinical use.
