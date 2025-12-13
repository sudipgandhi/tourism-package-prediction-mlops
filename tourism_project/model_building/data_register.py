import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

dataset_repo = "sudipgandhi/tourism-package-prediction"

local_data_path = "tourism_project/data"

# Check if repo exists, otherwise create it
try:
    api.repo_info(repo_id=dataset_repo, repo_type="dataset")
    print(f"Dataset repo '{dataset_repo}' already exists.")
except RepositoryNotFoundError:
    print("Dataset repo not found. Creating…")
    create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)

# Upload the data folder
api.upload_folder(
    folder_path=local_data_path,
    repo_id=dataset_repo,
    repo_type="dataset",
)

print("Data uploaded successfully.")
