"""
Hosting
--------
Pushes the deployment files (Streamlit app, Dockerfile, requirements.txt)
to a Hugging Face Space so that the model is publicly accessible via a
web front-end.
"""

import os
from huggingface_hub import HfApi

HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
SPACE_REPO = f"{HF_USERNAME}/tourism-package-app"
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Create the Space (Docker SDK, since we deploy via a Dockerfile)
api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)

files_to_push = ["app.py", "requirements.txt", "Dockerfile"]

for filename in files_to_push:
    api.upload_file(
        path_or_fileobj=f"tourism_project/deployment/{filename}",
        path_in_repo=filename,
        repo_id=SPACE_REPO,
        repo_type="space",
    )

print(f"App deployed at: https://huggingface.co/spaces/{SPACE_REPO}")
