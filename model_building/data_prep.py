"""
Data Preparation
-----------------
1. Loads the raw dataset directly from the Hugging Face dataset space.
2. Cleans the data (drops identifier columns, fixes inconsistent category
   labels, removes duplicate rows).
3. Splits the cleaned data into train / test sets and saves them locally.
4. Pushes the resulting train.csv / test.csv files back to the Hugging Face
   dataset space.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

# ---- Configuration ---------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
DATASET_REPO = f"{HF_USERNAME}/tourism-package-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")

TARGET_COL = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]
OUTPUT_DIR = "tourism_project/data"

# ---- 1. Load the dataset directly from the Hugging Face dataset space ----
raw_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(raw_path)
print("Raw data shape:", df.shape)

# ---- 2. Data cleaning --------------------------------------------------
# Drop identifier / unnamed index columns that carry no predictive signal
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Fix known inconsistent category labels
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

print("Cleaned data shape:", df.shape)
print(df.isnull().sum())

# ---- 3. Train / test split ---------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET_COL],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ---- 4. Push train/test back to the Hugging Face dataset space --------
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo="train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo="test.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

print("Train and test sets pushed to Hugging Face dataset space.")
