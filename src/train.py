import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report

# Configuration
HF_USERNAME = "iStillWaters"  # HuggingFace Profile
DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"
MODEL_REPO = f"{HF_USERNAME}/tourism-prediction-model"

def train_model():
    print("Loading processed data...")
    dataset = load_dataset(DATASET_REPO)
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    # --- ENCODING ---
    # We must save encoders to use them in the App later
    le_dict = {}
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    
    print("   Encoding categorical variables...")
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        # Use the same encoder for test, handle unseen items by ignoring/defaulting (simple approach)
        # For simplicity here, we assume test set is a subset of known categories
        test_df[col] = le.transform(test_df[col])
        le_dict[col] = le

    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']

    # --- MODEL TUNING ---
    models_config = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {'n_estimators': [50, 100], 'max_depth': [10, 20], 'class_weight': ['balanced']}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {'n_estimators': [50, 100], 'learning_rate': [0.1], 'scale_pos_weight': [3, 5]}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {'n_estimators': [50], 'learning_rate': [0.1]}
        }
    }

    best_recall = 0
    winner_name = ""
    winner_model = None

    print("Training and Tuning models...")
    for name, config in models_config.items():
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='recall', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        # Evaluate
        y_pred = grid.best_estimator_.predict(X_test)
        current_recall = recall_score(y_test, y_pred)
        
        print(f"   {name} -> Recall: {current_recall:.4f}")
        
        if current_recall > best_recall:
            best_recall = current_recall
            winner_name = name
            winner_model = grid.best_estimator_

    print(f"\nWINNER: {winner_name} (Recall: {best_recall:.4f})")
    
    # --- SAVE ARTIFACTS ---
    print("Saving artifacts and uploading to Hugging Face...")
    joblib.dump(winner_model, "model.joblib")
    joblib.dump(le_dict, "encoders.joblib")

    # Upload using HfApi
    api = HfApi()
    try:
        api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"   Repo creation warning: {e}")

    api.upload_file(path_or_fileobj="model.joblib", path_in_repo="model.joblib", repo_id=MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj="encoders.joblib", path_in_repo="encoders.joblib", repo_id=MODEL_REPO, repo_type="model")
    
    print("Training complete. Model uploaded.")

if __name__ == "__main__":
    train_model()
