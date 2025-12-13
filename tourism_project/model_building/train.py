import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ============================
# HF AUTH
# ============================
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# ============================
# LOAD DATA FROM HF DATASET
# ============================
DATASET_REPO = "sudipgandhi/tourism-package-prediction"

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

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# ============================
# MODELS
# ============================
models = {
    "xgboost": xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=42
    ),
}

results = {}
best_model = None
best_model_name = None
best_f1 = -1
best_threshold = 0.5

# ============================
# TRAIN & SELECT BEST MODEL
# ============================
for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(Xtrain, ytrain)

    probs = pipe.predict_proba(Xtest)[:, 1]

    # FIND BEST THRESHOLD
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(ytest, preds))

    max_f1 = max(f1_scores)
    opt_threshold = thresholds[np.argmax(f1_scores)]

    results[name] = {
        "best_f1": max_f1,
        "optimal_threshold": float(opt_threshold),
    }

    if max_f1 > best_f1:
        best_f1 = max_f1
        best_model = pipe
        best_model_name = name
        best_threshold = opt_threshold

# ============================
# SAVE ARTIFACTS
# ============================
joblib.dump(best_model, "best_model.joblib")

with open("best_threshold.json", "w") as f:
    json.dump(
        {"threshold": float(best_threshold), "model": best_model_name},
        f,
        indent=4
    )

with open("model_comparison_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# ============================
# UPLOAD TO HF MODEL HUB
# ============================
MODEL_REPO = "sudipgandhi/tourism-package-prediction-model"

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model")

for file in ["best_model.joblib", "best_threshold.json", "model_comparison_metrics.json"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=MODEL_REPO,
        repo_type="model",
    )

print(f"Best model: {best_model_name}")
print(f"Optimal threshold: {best_threshold:.2f}")
