#!/usr/bin/env python3

import pickle
from pathlib import Path

# Read feature list
with open('models/mortality/feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)

print("Feature list from trained model:")
for feature in feature_list:
    print(f"- {feature}") 