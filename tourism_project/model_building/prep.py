"""
DATA PREPARATION SCRIPT
----------------------
Purpose:
- Loads raw dataset from Hugging Face Dataset Hub
- Performs basic cleaning
- Splits data into train/test
- Uploads processed splits back to HF Dataset Hub
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ============================
# CONFIGURATION
# ============================
DATASET_REPO = "sudipgandhi/tourism-package-prediction"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found.")

api = HfApi(token=HF_TOKEN)

# ============================
# LOAD DATA FROM HF
# ============================
print("Loading dataset from Hugging Face...")
df = pd.read_csv(f"hf://datasets/{DATASET_REPO}/tourism.csv")

# ============================
# BASIC CLEANING
# ============================
# Drop ID / unnecessary columns if present
drop_cols = ["CustomerID", "Unnamed: 0"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Target variable
TARGET_COL = "ProdTaken"

if TARGET_COL not in df.columns:
    raise ValueError("Target column 'ProdTaken' not found.")

# ============================
# TRAIN / TEST SPLIT
# ============================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# SAVE SPLITS LOCALLY
# ============================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ============================
# UPLOAD SPLITS TO HF
# ============================
for file in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )

print("Data preparation completed and uploaded successfully.")
