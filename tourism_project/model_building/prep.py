import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

dataset_repo = "sudipgandhi/tourism-package-prediction"

# Load dataset directly from Hugging Face
df = pd.read_csv(f"hf://datasets/{dataset_repo}/tourism.csv")
print("Dataset loaded from HF dataset repo.")

# Drop unnecessary columns
drop_cols = ["CustomerID", "Unnamed: 0"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target column
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload split files
for f in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id=dataset_repo,
        repo_type="dataset",
    )

print("Train/Test split uploaded successfully.")
