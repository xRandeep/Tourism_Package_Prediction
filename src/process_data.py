import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
import os

# Configuration
HF_USERNAME = "iStillWaters"  # HuggingFace Profile
DATASET_REPO = f"{HF_USERNAME}/tourism-project-data"

def process_data():
    print("Loading data from Hugging Face...")
    # Load raw data (Assuming you uploaded it as 'raw' config or just the main dataset)
    # If this fails, ensure you ran the "Registration" step one-time script first
    dataset = load_dataset(DATASET_REPO, split="train") 
    df = dataset.to_pandas()

    print(f"   Original Shape: {df.shape}")

    # --- CLEANING STEPS ---
    # 1. Drop ID columns
    df = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')

    # 2. Fix Gender: 'Fe Male' -> 'Female'
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    # 3. Fix Occupation: 'Free Lancer' -> 'Small Business'
    df['Occupation'] = df['Occupation'].replace('Free Lancer', 'Small Business')
    
    # 4. Handle Missing Values (Simple Drop for this pipeline)
    df = df.dropna()

    print(f"   Cleaned Shape: {df.shape}")

    # --- SPLITTING ---
    # Stratify by ProdTaken to handle class imbalance
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'])

    # --- UPLOAD BACK TO HF ---
    print("Uploading processed splits to Hugging Face...")
    
    # Convert back to HF Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create a DatasetDict
    dd = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Push to Hub (using the HF_TOKEN from environment variables)
    # This works automatically in GitHub Actions if the Secret is set
    dd.push_to_hub(DATASET_REPO)
    
    print("Data processing complete. Splits uploaded.")

if __name__ == "__main__":
    process_data()
