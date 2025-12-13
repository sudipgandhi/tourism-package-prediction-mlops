import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="sudipgandhi/tourism-package-prediction-space",
    repo_type="space",
)

print("Deployment folder uploaded to HF Space.")
