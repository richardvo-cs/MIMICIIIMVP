#!/usr/bin/env python3

import pickle
import pandas as pd
from pathlib import Path

def inspect_features(model_name):
    # Load features
    feature_path = Path(f"data/features/{model_name}_features.pkl")
    with open(feature_path, 'rb') as f:
        features_df = pickle.load(f)
    
    print(f"\nFeatures for {model_name} model:")
    print("\nColumns:")
    for col in features_df.columns:
        print(f"- {col}")
    
    print("\nSample data types:")
    print(features_df.dtypes)
    
    print("\nSample unique values for categorical columns:")
    categorical_columns = features_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"\n{col}:")
        print(features_df[col].unique())

# Inspect features for mortality model only
try:
    inspect_features('mortality')
except Exception as e:
    print(f"Error inspecting mortality features: {str(e)}") 