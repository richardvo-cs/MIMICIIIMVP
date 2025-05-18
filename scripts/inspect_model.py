#!/usr/bin/env python3

import pickle
from joblib import load
from pathlib import Path

def inspect_model(model_name):
    print(f"\nInspecting {model_name} model:")
    
    # Load model
    model_path = Path(f"models/{model_name}/random_forest_model.joblib")
    if not model_path.exists():
        model_path = Path(f"models/{model_name}/gradient_boosting_model.joblib")
    
    if not model_path.exists():
        print(f"No model found for {model_name}")
        return
    
    model = load(model_path)
    
    # Get feature names
    if hasattr(model, 'feature_names_in_'):
        print("\nFeature names used during training:")
        for name in model.feature_names_in_:
            print(f"- {name}")
    else:
        print("Model does not have feature_names_in_ attribute")

# Inspect mortality model only
inspect_model('mortality') 