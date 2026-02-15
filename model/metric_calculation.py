# Import metrics
try:
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import matthews_corrcoef
except ImportError as e:
    print(f"Import error: {e}")
    raise


# Function to calculate all evaluation metrics
def calculate_metrics(y_test, y_pred, y_pred_proba=None):
    """
    Calculate all evaluation metrics for a classification model.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels (binary: 0 or 1)
    y_pred_proba : array-like, optional
        Predicted probabilities (needed for AUC score)
    
    Returns:
    --------
    dict : Dictionary containing all 6 metrics
    """
    metrics = {}
    
    # 1. Accuracy
    metrics['Accuracy'] = accuracy_score(y_test, y_pred)
    
    # 2. AUC Score
    if y_pred_proba is not None:
        metrics['AUC Score'] = roc_auc_score(y_test, y_pred_proba)
    else:
        metrics['AUC Score'] = None  # Will be calculated if probabilities are available
    
    # 3. Precision
    metrics['Precision'] = precision_score(y_test, y_pred, zero_division=0)

    # 4. Recall
    metrics['Recall'] = recall_score(y_test, y_pred, zero_division=0)

    # 5. F1 Score
    metrics['F1 Score'] = f1_score(y_test, y_pred, zero_division=0)

    # 6. Matthews Correlation Coefficient (MCC)
    metrics['MCC Score'] = matthews_corrcoef(y_test, y_pred)

    return metrics

# Function to display metrics in a formatted way
def display_metrics(model_name, metrics):
    """
    Display metrics in a formatted table format.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    metrics : dict
        Dictionary of metrics returned by calculate_metrics()
    """
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name:<25} : {value:.4f}")
        else:
            print(f"{metric_name:<25} : Not Available")
    print(f"{'='*50}\n")