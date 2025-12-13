"""
MODEL TRAINING SCRIPT
---------------------
Purpose:
- Loads prepared train/test data from HF
- Trains multiple ML models with preprocessing pipeline
- Selects best model using F1-score
- Learns optimal probability threshold
- Saves and uploads model artifacts to HF Model Hub
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ============================
# CONFIGURATION
# ============================
DATASET_REPO = "sudipgandhi/tourism-package-prediction"
MODEL_REPO = "sudipgandhi/tourism-package-prediction-model"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found.")

api = HfApi(token=HF_TOKEN)

# ============================
# LOAD TRAIN / TEST DATA
# ============================
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()

# ============================
# FEATURE GROUPS
# ============================
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

numeric_features = [
    "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

# ============================
# PREPROCESSING PIPELINE
# ============================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# ============================
# MODELS TO COMPARE
# ============================
models = {
    "XGB": xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        random_state=42, class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    )
}

best_f1 = -1
best_model = None
best_model_name = None

# ============================
# TRAIN & EVALUATE
# ============================
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(Xtrain, ytrain)
    preds = pipeline.predict(Xtest)

    f1 = f1_score(ytest, preds)
    print(f"{name} F1-score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# ============================
# THRESHOLD SELECTION
# ============================
probs = best_model.predict_proba(Xtrain)[:, 1]
thresholds = np.linspace(0.1, 0.9, 50)

best_threshold = 0.5
best_f1_thresh = 0

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(ytrain, preds)
    if f1 > best_f1_thresh:
        best_f1_thresh = f1
        best_threshold = t

# ============================
# SAVE ARTIFACTS
# ============================
joblib.dump(best_model, "best_model.joblib")

with open("best_threshold.json", "w") as f:
    json.dump({"threshold": best_threshold}, f)

# ============================
# UPLOAD TO HF MODEL HUB
# ============================
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

for file in ["best_model.joblib", "best_threshold.json"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=MODEL_REPO,
        repo_type="model"
    )

print("Model training and upload completed.")
