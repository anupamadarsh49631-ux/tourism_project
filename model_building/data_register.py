"""
Data Registration
------------------
Uploads the raw tourism.csv file to a Hugging Face Hub dataset repository so
that it becomes the single source of truth for the rest of the pipeline.

Requires an environment variable HF_TOKEN with write access to the
Hugging Face Hub. Update HF_USERNAME / DATASET_REPO below to match your
own Hugging Face account before running.
"""

import os
from huggingface_hub import HfApi

# ---- Configuration (update with your own Hugging Face username) ----------
HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
DATASET_REPO = f"{HF_USERNAME}/tourism-package-dataset"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Create the dataset repo if it doesn't already exist
api.create_repo(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    private=False,
    exist_ok=True,
)

# Upload the raw data file
api.upload_file(
    path_or_fileobj=LOCAL_DATA_PATH,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

print(f"Raw dataset registered at: https://huggingface.co/datasets/{DATASET_REPO}")
