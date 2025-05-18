#!/usr/bin/env python3

import pickle
from pathlib import Path

def inspect_scaler(model_name):
    print(f"\nInspecting {model_name} scaler:")
    
    # Load scaler
    scaler_path = Path(f"models/{model_name}/scaler.pkl")
    if not scaler_path.exists():
        print(f"No scaler found for {model_name}")
        return
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("\nScaler type:", type(scaler))
    if hasattr(scaler, 'feature_names_in_'):
        print("\nFeature names used during scaling:")
        for name in scaler.feature_names_in_:
            print(f"- {name}")
    else:
        print("Scaler does not have feature_names_in_ attribute")

# Inspect mortality scaler only
inspect_scaler('mortality') 