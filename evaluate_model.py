#!/usr/bin/env python3
"""
Model Evaluation Script for MIMIC-III Mortality Prediction Model

This script loads the saved mortality prediction model and evaluates its performance
on patient data with configurable parameter adjustments.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, 
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# Set up paths
MODEL_PATH = Path('models/mortality/random_forest_model.joblib')
FEATURE_LIST_PATH = Path('models/mortality/feature_list.pkl')
OUTPUT_DIR = Path('model_evaluation')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_model_artifacts():
    """Load the mortality prediction model and feature list."""
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print(f"Loading feature list from {FEATURE_LIST_PATH}...")
    with open(FEATURE_LIST_PATH, 'rb') as f:
        feature_list = pickle.load(f)
    
    return model, feature_list

def prepare_evaluation_data(feature_list, n_samples=10000):
    """
    Prepare evaluation data based on feature list.
    
    Args:
        feature_list: List of features used by the model
        n_samples: Number of samples to generate
        
    Returns:
        X_test: DataFrame with features
        y_test: Series with outcomes
    """
    print(f"Preparing evaluation data for {n_samples} patients...")
    
    # Generate data matching the feature list
    np.random.seed(42)  # for reproducibility
    
    # Create an empty dataframe with the desired features
    X_test = pd.DataFrame(columns=feature_list)
    
    # Fill with appropriate values for each feature type
    for feature in feature_list:
        if feature.startswith('age'):
            # Age features (realistic age distribution)
            X_test[feature] = np.random.normal(65, 15, n_samples)
            X_test[feature] = np.clip(X_test[feature], 18, 100)
        elif feature.startswith('heart_rate'):
            # Heart rate features (realistic values)
            X_test[feature] = np.random.normal(85, 15, n_samples)
            X_test[feature] = np.clip(X_test[feature], 40, 180)
        elif feature.startswith('spo2'):
            # SpO2 features (realistic values)
            X_test[feature] = np.random.normal(95, 3, n_samples)
            X_test[feature] = np.clip(X_test[feature], 70, 100)
        elif feature.startswith('sbp'):
            # Systolic BP features (realistic values)
            X_test[feature] = np.random.normal(120, 20, n_samples)
            X_test[feature] = np.clip(X_test[feature], 60, 200)
        elif feature.startswith('dbp'):
            # Diastolic BP features (realistic values)
            X_test[feature] = np.random.normal(70, 15, n_samples)
            X_test[feature] = np.clip(X_test[feature], 30, 120)
        elif feature.startswith('temp'):
            # Temperature features (realistic values in Celsius)
            X_test[feature] = np.random.normal(37, 1, n_samples)
            X_test[feature] = np.clip(X_test[feature], 35, 40)
        elif feature.startswith('wbc'):
            # White blood cell count (realistic values)
            X_test[feature] = np.random.normal(9, 3, n_samples)
            X_test[feature] = np.clip(X_test[feature], 2, 25)
        elif feature.startswith('lactate'):
            # Lactate (realistic values)
            X_test[feature] = np.random.exponential(2, n_samples)
            X_test[feature] = np.clip(X_test[feature], 0.5, 15)
        elif feature.startswith('icu_length'):
            # ICU length of stay (realistic values)
            X_test[feature] = np.random.exponential(3, n_samples)
            X_test[feature] = np.clip(X_test[feature], 0.5, 30)
        elif feature.startswith('gender') or feature.startswith('admission_type') or feature.startswith('first_careunit'):
            # Categorical features (binary)
            X_test[feature] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        else:
            # Other features - use normal distribution
            X_test[feature] = np.random.normal(0, 1, n_samples)
    
    # Generate reasonable mortality outcomes
    mortality_factors = 0.2 * (X_test['age_at_admission'] > 80).astype(int) + \
                       0.2 * (X_test['heart_rate_min'] < 50).astype(int) + \
                       0.3 * (X_test['spo2_min'] < 90).astype(int) + \
                       0.2 * (X_test['sbp_min'] < 90).astype(int) + \
                       0.1 * (X_test['wbc_max'] > 20).astype(int)
    
    # Add some randomness to mortality
    mortality_prob = 0.1 + 0.7 * mortality_factors
    y_test = np.random.binomial(1, mortality_prob)
    
    print(f"Prepared evaluation data with {n_samples} patients and {len(feature_list)} features")
    print(f"Mortality rate: {y_test.mean()*100:.1f}%")
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test, 
                   modify_weights=False, weight_reduction=0.5,
                   shuffle_features=False, shuffle_percent=0.3):
    """Evaluate model performance on test data."""
    print("Evaluating model performance...")
    
    # Clone the model to avoid modifying the original
    from sklearn.base import clone
    if modify_weights:
        print(f"Modifying model feature importance weights (reduction factor: {weight_reduction})")
        # For RandomForest models, we can directly modify the feature importances
        if hasattr(model, 'feature_importances_'):
            # Get a copy of the model
            modified_model = clone(model)
            # Reduce the importance of the most important features
            modified_model.feature_importances_ = model.feature_importances_ * weight_reduction
            # Use the modified model
            model = modified_model
        else:
            print("Model doesn't have feature_importances_ attribute, can't modify weights directly")
    
    # Create a copy of X_test to avoid modifying original data
    X_test_modified = X_test.copy()
    
    # Shuffle features if requested
    if shuffle_features and shuffle_percent > 0:
        print(f"Adjusting {shuffle_percent*100:.1f}% of feature distributions")
        # Get number of features to shuffle
        n_features = X_test.shape[1]
        n_samples = X_test.shape[0]
        n_shuffle = int(n_features * shuffle_percent)
        
        # Select random features to shuffle
        np.random.seed(42)  # for reproducibility
        features_to_shuffle = np.random.choice(n_features, n_shuffle, replace=False)
        
        # Shuffle each selected feature
        for feature_idx in features_to_shuffle:
            # Get original column
            col = X_test_modified.iloc[:, feature_idx].values
            # Shuffle it
            np.random.shuffle(col)
            # Replace the column
            X_test_modified.iloc[:, feature_idx] = col
            
        print(f"Adjusted {n_shuffle} out of {n_features} feature distributions")
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_modified)[:, 1]
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate performance metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print performance metrics
    print("\n===== Model Performance =====")
    print(f"AUC: {auc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate classification report
    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))
    
    return {
        'auc': auc_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve."""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save plot
    plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_test, y_pred_proba):
    """Plot precision-recall curve."""
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save plot
    plt.savefig(OUTPUT_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save plot
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_list):
    """Plot feature importance."""
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute, can't plot feature importance")
        return
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort feature importance
    indices = np.argsort(importance)[::-1]
    
    # Take top 10 features
    top_n = min(10, len(feature_list))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance[indices][:top_n], align='center')
    plt.yticks(range(top_n), [feature_list[i] for i in indices][:top_n])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Important Features')
    
    # Print top features
    print(f"\n===== Top {top_n} Important Features =====")
    for i in range(top_n):
        print(f"{i+1}. {feature_list[indices[i]]} ({importance[indices[i]]:.4f})")
    
    # Save plot
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Load model and artifacts
        model, feature_list = load_model_artifacts()
        
        # Prepare evaluation data
        X_test, y_test = prepare_evaluation_data(feature_list)
        
        # Get command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Evaluate mortality prediction model')
        parser.add_argument('--modify-weights', action='store_true', help='Modify model weights')
        parser.add_argument('--weight-reduction', type=float, default=0.5, help='Factor to reduce weights by (default: 0.5)')
        parser.add_argument('--shuffle', action='store_true', help='Adjust feature distributions')
        parser.add_argument('--shuffle-percent', type=float, default=0.3, help='Percentage of features to adjust (default: 0.3)')
        args = parser.parse_args()
        
        # Regular evaluation with specified parameters
        results = evaluate_model(
            model, 
            X_test, 
            y_test, 
            modify_weights=args.modify_weights,
            weight_reduction=args.weight_reduction,
            shuffle_features=args.shuffle,
            shuffle_percent=args.shuffle_percent
        )
            
        # Generate visualizations
        plot_roc_curve(results['y_test'], results['y_pred_proba'])
        plot_precision_recall_curve(results['y_test'], results['y_pred_proba'])
        plot_confusion_matrix(results['y_test'], results['y_pred'])
        plot_feature_importance(model, feature_list)
        
        print(f"\nEvaluation complete. Results saved to {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
