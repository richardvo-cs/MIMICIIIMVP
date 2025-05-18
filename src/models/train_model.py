#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIMIC-III Model Training Script
This script trains machine learning models to predict patient outcomes.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, roc_auc_score, precision_recall_curve, 
                             average_precision_score)

def load_feature_matrix(input_path='../../data/features/feature_matrix.csv'):
    """Load the feature matrix from a CSV file."""
    print(f"Loading feature matrix from {input_path}")
    return pd.read_csv(input_path)

def prepare_data_for_modeling(df, outcome_col='mortality_outcome', test_size=0.2, val_size=0.25):
    """
    Prepare data for modeling by separating features and target,
    and splitting into train, validation, and test sets.
    
    Args:
        df: Feature matrix DataFrame
        outcome_col: Name of the outcome column
        test_size: Proportion of data to use for testing
        val_size: Proportion of non-test data to use for validation
        
    Returns:
        X_train, X_val, X_test: Feature DataFrames for training, validation, and testing
        y_train, y_val, y_test: Target series for training, validation, and testing
    """
    # Remove rows with NaN in the outcome column
    df = df.dropna(subset=[outcome_col])
    
    # Define exclusion columns (not used as features)
    exclude_cols = [
        'subject_id', 'hadm_id', 'icustay_id', 'charttime', 
        'deathtime', 'dischtime', 'time_until_death', 'time_until_discharge'
    ]
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    # Add the outcome column to exclusion list
    exclude_cols.append(outcome_col)
    
    # Split features and target
    X = df.drop(columns=exclude_cols)
    y = df[outcome_col]
    
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessing_pipeline(X_train):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Args:
        X_train: Training feature DataFrame
        
    Returns:
        ColumnTransformer for preprocessing features
    """
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing steps for different column types
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def train_and_evaluate_models(X_train, X_val, y_train, y_val, preprocessor):
    """
    Train and evaluate machine learning models.
    
    Args:
        X_train: Training feature DataFrame
        X_val: Validation feature DataFrame
        y_train: Training target Series
        y_val: Validation target Series
        preprocessor: ColumnTransformer for preprocessing features
        
    Returns:
        Dictionary of trained models and their performance metrics
    """
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Parameter grids for hyperparameter tuning
    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        
        # Create a pipeline with preprocessing and the classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions on validation set
        y_val_pred = best_model.predict(X_val)
        y_val_prob = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate performance metrics
        val_roc_auc = roc_auc_score(y_val, y_val_prob)
        val_avg_precision = average_precision_score(y_val, y_val_prob)
        
        print(f"{name} Validation ROC AUC: {val_roc_auc:.4f}")
        print(f"{name} Validation Average Precision: {val_avg_precision:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred))
        
        # Store results
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'val_roc_auc': val_roc_auc,
            'val_avg_precision': val_avg_precision,
            'y_val_pred': y_val_pred,
            'y_val_prob': y_val_prob
        }
    
    return results

def plot_roc_curves(y_val, results, output_dir='../../reports/figures'):
    """
    Plot ROC curves for all models.
    
    Args:
        y_val: Validation target Series
        results: Dictionary of model results
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each model
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_val, result['y_val_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['val_roc_auc']:.3f})")
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(y_val, results, output_dir='../../reports/figures'):
    """
    Plot precision-recall curves for all models.
    
    Args:
        y_val: Validation target Series
        results: Dictionary of model results
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot precision-recall curve for each model
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_val, result['y_val_prob'])
        plt.plot(recall, precision, label=f"{name} (AP = {result['val_avg_precision']:.3f})")
    
    # Plot random classifier
    plt.axhline(y=y_val.mean(), color='k', linestyle='--', label=f'Random Classifier (AP = {y_val.mean():.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_on_test_set(X_test, y_test, best_model, output_dir='../../reports'):
    """
    Evaluate the best model on the test set.
    
    Args:
        X_test: Test feature DataFrame
        y_test: Test target Series
        best_model: Best trained model
        output_dir: Directory to save the evaluation report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions on test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    test_roc_auc = roc_auc_score(y_test, y_test_prob)
    test_avg_precision = average_precision_score(y_test, y_test_prob)
    
    print("\nTest Set Evaluation:")
    print(f"ROC AUC: {test_roc_auc:.4f}")
    print(f"Average Precision: {test_avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'figures', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save test set evaluation results
    test_results = {
        'roc_auc': test_roc_auc,
        'avg_precision': test_avg_precision,
        'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    # Save evaluation results to file
    with open(os.path.join(output_dir, 'test_evaluation.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    return test_results

def save_model(model, model_dir='../../models'):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_dir: Directory to save the model
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {os.path.join(model_dir, 'best_model.pkl')}")

def define_intervention_thresholds(y_val, y_val_prob, output_dir='../../reports'):
    """
    Define risk thresholds for clinical interventions.
    
    Args:
        y_val: Validation target Series
        y_val_prob: Predicted probabilities for validation set
        output_dir: Directory to save the threshold report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate precision and recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    
    # Convert to DataFrame for easier analysis
    thresholds_df = pd.DataFrame({
        'threshold': np.append(thresholds, 1.0),  # Add 1.0 as the last threshold
        'precision': precision,
        'recall': recall
    })
    
    # Find the threshold that gives at least 80% precision (high certainty of deterioration)
    high_risk_threshold = thresholds_df[thresholds_df['precision'] >= 0.8]['threshold'].min()
    
    # Find the threshold that gives at least 80% recall (captures most deteriorations)
    medium_risk_threshold = thresholds_df[thresholds_df['recall'] >= 0.8]['threshold'].max()
    
    # Define low risk as below medium risk
    low_risk_threshold = medium_risk_threshold
    
    # Create threshold report
    threshold_report = {
        'high_risk_threshold': high_risk_threshold,
        'medium_risk_threshold': medium_risk_threshold,
        'low_risk_threshold': low_risk_threshold,
        'thresholds_data': thresholds_df.to_dict()
    }
    
    # Save threshold report to file
    with open(os.path.join(output_dir, 'intervention_thresholds.pkl'), 'wb') as f:
        pickle.dump(threshold_report, f)
    
    print("\nIntervention Thresholds:")
    print(f"High Risk (≥ {high_risk_threshold:.3f}): Schedule immediate clinician visit")
    print(f"Medium Risk ({medium_risk_threshold:.3f} - {high_risk_threshold:.3f}): Increase monitoring frequency")
    print(f"Low Risk (< {medium_risk_threshold:.3f}): Standard monitoring")
    
    # Create a visualization of the thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_df['threshold'], thresholds_df['precision'], label='Precision')
    plt.plot(thresholds_df['threshold'], thresholds_df['recall'], label='Recall')
    plt.axvline(x=high_risk_threshold, color='r', linestyle='--', 
                label=f'High Risk Threshold ({high_risk_threshold:.3f})')
    plt.axvline(x=medium_risk_threshold, color='g', linestyle='--', 
                label=f'Medium Risk Threshold ({medium_risk_threshold:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'figures', 'intervention_thresholds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return threshold_report

def main():
    """Main function to train and evaluate models."""
    # Load data
    feature_matrix = load_feature_matrix()
    
    # Prepare data for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(feature_matrix)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X_train, X_val, y_train, y_val, preprocessor)
    
    # Plot ROC curves
    plot_roc_curves(y_val, model_results)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(y_val, model_results)
    
    # Find the best model based on validation ROC AUC
    best_model_name = max(model_results, key=lambda k: model_results[k]['val_roc_auc'])
    best_model = model_results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model parameters: {model_results[best_model_name]['best_params']}")
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(X_test, y_test, best_model)
    
    # Define intervention thresholds
    threshold_report = define_intervention_thresholds(
        y_val, model_results[best_model_name]['y_val_prob']
    )
    
    # Save the best model
    save_model(best_model)
    
    print("\nModel training and evaluation complete.")

if __name__ == "__main__":
    main() 