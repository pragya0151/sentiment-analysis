# ============================================================
# Download model from HuggingFace
# Run this before starting the app: python download_model.py
# ============================================================

from huggingface_hub import hf_hub_download, snapshot_download
import os

print("Downloading sentiment model from HuggingFace...")

os.makedirs("models", exist_ok=True)

# Download full model folder
snapshot_download(
    repo_id="pragya0151/sentiment-analyzer",
    repo_type="space",
    local_dir="models/sentiment_model",
    ignore_patterns=["*.py", "*.txt", "*.md"]
)

print("Model downloaded successfully to models/")
print("You can now run: streamlit run app.py")