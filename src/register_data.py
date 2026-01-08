import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import login, HfApi

# Configuration
HF_USERNAME = "iStillWaters"  # HuggingFace profile
DATASET_REPO_NAME = "tourism-package-data"
REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
RAW_DATA_PATH = "data/tourism.csv"

def register_data():
    # 1. Authentication
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN found. Logging in...")
        login(token=hf_token)
    else:
        raise ValueError("❌ HF_TOKEN not found in environment variables.")

    # 2. Check if local file exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Could not find raw data at: {RAW_DATA_PATH}")
    
    print(f"mn Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Shape: {df.shape}")

    # 3. Convert to Hugging Face Dataset
    # We label this split 'train' by default as it's the raw source
    raw_dataset = Dataset.from_pandas(df)

    # 4. Create Repo (if it doesn't exist) & Upload
    api = HfApi()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"   Repository '{REPO_ID}' is ready.")
    except Exception as e:
        print(f"   Repo check warning: {e}")

    print(f"Uploading raw data to {REPO_ID}...")
    raw_dataset.push_to_hub(REPO_ID)
    print("✅ Data Registration Complete.")

if __name__ == "__main__":
    register_data()
