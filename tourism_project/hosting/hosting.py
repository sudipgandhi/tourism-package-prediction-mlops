"""
HOSTING SCRIPT
--------------
Pushes Streamlit deployment files
to the Hugging Face Space.
"""

import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN is missing.")

api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="sudipgandhi/tourism-package-prediction-space",
    repo_type="space"
)

print("Deployment files successfully pushed to Hugging Face Space.")
