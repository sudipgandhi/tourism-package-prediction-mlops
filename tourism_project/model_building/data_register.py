"""
DATA REGISTRATION SCRIPT
------------------------
Purpose:
- Registers the raw tourism dataset on Hugging Face Dataset Hub
- Uploads local CSV files from GitHub repo to HF dataset repo
- This script is executed via GitHub Actions

Prerequisites:
- HF_TOKEN must be set as an environment variable
- Dataset repo must exist OR will be created automatically
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ============================
# CONFIGURATION
# ============================
DATASET_REPO_ID = "sudipgandhi/tourism-package-prediction"
REPO_TYPE = "dataset"
LOCAL_DATA_FOLDER = "tourism_project/data"

# ============================
# AUTHENTICATION
# ============================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Please add it to GitHub Secrets.")

api = HfApi(token=HF_TOKEN)

# ============================
# CHECK / CREATE DATASET REPO
# ============================
try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE)
    print("Dataset repository already exists.")
except RepositoryNotFoundError:
    print("Dataset repository not found. Creating one...")
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
        private=False
    )
    print("Dataset repository created.")

# ============================
# UPLOAD LOCAL DATA FOLDER
# ============================
print("Uploading dataset files to Hugging Face...")
api.upload_folder(
    folder_path=LOCAL_DATA_FOLDER,
    repo_id=DATASET_REPO_ID,
    repo_type=REPO_TYPE
)

print("Dataset upload completed successfully.")
