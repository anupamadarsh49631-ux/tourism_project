"""
Model Training and Registration with Experimentation Tracking
---------------------------------------------------------------
1. Loads the train / test data from the Hugging Face dataset space.
2. Builds a preprocessing + model pipeline for several candidate
   algorithms (Decision Tree, Bagging, Random Forest, AdaBoost,
   Gradient Boosting, XGBoost).
3. Tunes each candidate with GridSearchCV and logs every run
   (parameters + metrics + the fitted model) with MLflow.
4. Evaluates all candidates on the held-out test set.
5. Registers the best-performing model on the Hugging Face model hub.
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ---- Configuration ---------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
DATASET_REPO = f"{HF_USERNAME}/tourism-package-dataset"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"
HF_TOKEN = os.getenv("HF_TOKEN")
TARGET_COL = "ProdTaken"

# ---- 1. Load train / test data from the Hugging Face dataset space -----
train_path = hf_hub_download(repo_id=DATASET_REPO, filename="train.csv", repo_type="dataset", token=HF_TOKEN)
test_path = hf_hub_download(repo_id=DATASET_REPO, filename="test.csv", repo_type="dataset", token=HF_TOKEN)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
)

# ---- 2. Define candidate models + hyperparameter grids -----------------
candidates = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"model__max_depth": [4, 6, 8, None], "model__min_samples_split": [2, 5, 10]},
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {"model__n_estimators": [50, 100], "model__max_samples": [0.7, 1.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"model__n_estimators": [100, 200], "model__max_depth": [6, 10, None]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"model__n_estimators": [50, 100], "model__learning_rate": [0.5, 1.0]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="logloss"),
        {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
    ),
}

# ---- 3. Tune + track each candidate with MLflow -------------------------
mlflow.set_experiment("tourism-package-prediction")

results = []
best_model_name, best_f1, best_pipeline = None, -1.0, None

for name, (estimator, param_grid) in candidates.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)

        # Log all tuned parameters
        mlflow.log_params(grid.best_params_)

        # ---- 4. Evaluate model performance on the held-out test set ----
        preds = grid.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
        except Exception:
            auc = np.nan

        mlflow.log_metrics(
            {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
        )
                mlflow.sklearn.log_model(grid.best_estimator_, name, serialization_format="pickle")

        results.append(
            {"model": name, "best_params": grid.best_params_, "accuracy": acc,
             "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
        )
        print(f"{name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} roc_auc={auc:.4f}")

        if f1 > best_f1:
            best_f1, best_model_name, best_pipeline = f1, name, grid.best_estimator_

results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
print("\nModel comparison (sorted by F1):")
print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")

# ---- 5. Register the best model on the Hugging Face model hub ----------
os.makedirs("tourism_project/model_building/artifacts", exist_ok=True)
model_path = "tourism_project/model_building/artifacts/best_model.joblib"
joblib.dump(best_pipeline, model_path)

results_csv_path = "tourism_project/model_building/artifacts/model_comparison.csv"
results_df.to_csv(results_csv_path, index=False)

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib", repo_id=MODEL_REPO, repo_type="model")
api.upload_file(path_or_fileobj=results_csv_path, path_in_repo="model_comparison.csv", repo_id=MODEL_REPO, repo_type="model")

print(f"Best model registered at: https://huggingface.co/{MODEL_REPO}")
