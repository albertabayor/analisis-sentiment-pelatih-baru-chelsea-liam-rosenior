"""
Sentiment Analysis Project - Chelsea Liam Rosenior Appointment

Machine learning model training and evaluation utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import joblib
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def train_logistic_regression(X_train, y_train, params=None):
    """
    Train Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params (dict): Hyperparameters for GridSearchCV. If None, uses defaults.
    
    Returns:
        tuple: (best_model, grid_search_results)
    """
    print("\n=== Training Logistic Regression ===\n")
    
    # Default hyperparameters
    # Note: Using 'lbfgs' solver which supports multiclass classification
    if params is None:
        params = {
            'C': [0.1, 1, 10],
            'max_iter': [100, 200],
            'solver': ['lbfgs'],  # 'lbfgs' supports multiclass
            'random_state': [42]
        }
    
    # Initialize model - lbfgs handles multiclass automatically
    lr = LogisticRegression()
    
    # Grid search
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        lr, 
        params, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return best_model, grid_search


def train_naive_bayes(X_train, y_train, params=None):
    """
    Train Multinomial Naive Bayes model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params (dict): Hyperparameters for GridSearchCV. If None, uses defaults.
    
    Returns:
        tuple: (best_model, grid_search_results)
    """
    print("\n=== Training Multinomial Naive Bayes ===\n")
    
    # Default hyperparameters
    if params is None:
        params = {
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        }
    
    # Initialize model
    nb = MultinomialNB()
    
    # Grid search
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        nb, 
        params, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return best_model, grid_search


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"\n=== Evaluating {model_name} ===\n")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        save_path (str): Path to save the plot. If None, doesn't save.
    
    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()
    plt.close()
    
    return fig


def compare_models(metrics_list):
    """
    Compare multiple models and create comparison table.
    
    Args:
        metrics_list (list): List of metric dictionaries
    
    Returns:
        pd.DataFrame: Comparison table
    """
    print("\n=== Model Comparison ===\n")
    
    # Create dataframe
    comparison_df = pd.DataFrame(metrics_list)
    
    # Sort by accuracy (descending)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    # Format metrics
    for col in ['accuracy', 'precision', 'recall', 'f1_score']:
        comparison_df[col] = comparison_df[col].round(4)
    
    # Print table
    print(comparison_df.to_string(index=False))
    
    # Highlight best model
    best_model = comparison_df.iloc[0]
    print(f"\n✓ Best Model: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})")
    
    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    """
    Plot model comparison bar chart.
    
    Args:
        comparison_df (pd.DataFrame): Model comparison dataframe
        save_path (str): Path to save the plot. If None, doesn't save.
    
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Melt dataframe for easier plotting
    df_melted = comparison_df.melt(
        id_vars=['model'], 
        value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
        var_name='metric',
        value_name='score'
    )
    
    # Create subplots for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        metric_data = df_melted[df_melted['metric'] == metric]
        
        sns.barplot(
            data=metric_data,
            x='model',
            y='score',
            hue='model',
            legend=False,
            ax=ax,
            palette='viridis'
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        
        # Add value labels on bars
        for i, v in enumerate(metric_data['score']):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to: {save_path}")
    
    plt.show()
    plt.close()
    
    return fig


def save_model(model, model_name, save_dir=None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name (str): Name of the model (without extension)
        save_dir (str): Directory to save model. If None, uses current directory.
    """
    if save_dir is None:
        save_dir = os.getcwd()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    print(f"✓ Model saved to: {model_path}")


def load_model(model_path):
    """
    Load saved model from disk.
    
    Args:
        model_path (str): Path to model file
    
    Returns:
        Loaded model
    """
    model = joblib.load(model_path)
    print(f"✓ Model loaded from: {model_path}")
    return model


def save_metrics(metrics_list, save_dir=None, filename='metrics.json'):
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics_list (list): List of metric dictionaries
        save_dir (str): Directory to save metrics. If None, uses current directory.
        filename (str): Name of metrics file
    """
    import json
    from src.utils import get_outputs_path
    
    if save_dir is None:
        save_dir = get_outputs_path('metrics')
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_metrics = []
    for metrics in metrics_list:
        json_metrics.append({
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        })
    
    # Save to JSON
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"✓ Metrics saved to: {file_path}")


if __name__ == "__main__":
    # Test model functions
    print("Testing model functions...")
    
    # Create sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    lr_model, lr_results = train_logistic_regression(X_train, y_train)
    nb_model, nb_results = train_naive_bayes(X_train, y_train)
    
    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    nb_metrics = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    
    # Compare models
    comparison_df = compare_models([lr_metrics, nb_metrics])
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, lr_model.predict(X_test), "Logistic Regression")
