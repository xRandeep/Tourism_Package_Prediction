import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
import os
from huggingface_hub import login

# Configuration
HF_USERNAME = "iStillWaters" 
DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"

def process_data():
    # Authentication
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Loading RAW data from {DATASET_REPO}...")
    
    # --- FIX 1: Load specific "raw" config ---
    try:
        dataset = load_dataset(DATASET_REPO, "raw", split="train") 
    except Exception as e:
        print(f"❌ Error loading raw data. Ensure register_data.py ran successfully.")
        raise e
        
    df = dataset.to_pandas()
    print(f"   Original Shape: {df.shape}")

    # --- CLEANING ---
    df = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    df['Occupation'] = df['Occupation'].replace('Free Lancer', 'Small Business')
    df = df.dropna()

    # --- SPLITTING ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'])

    # --- UPLOAD ---
    print("Uploading PROCESSED splits to Hugging Face...")
    
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    dd = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # --- FIX 2: Push to "processed" config ---
    dd.push_to_hub(DATASET_REPO, config_name="processed")
    
    print("✅ Data processing complete. Uploaded to 'processed' config.")

if __name__ == "__main__":
    process_data()