# Trained model files

Place the trained model and labels here for Docker Compose deployment:

```text
models/
  skin_disease_cnn.keras
  class_names.json
```

Train from the project root using `python train.py --data dataset --output models`. Model artifacts are ignored by Git and excluded from the Docker build context. Never commit patient images, training data, or model weights unless you have verified their license and intended distribution.
