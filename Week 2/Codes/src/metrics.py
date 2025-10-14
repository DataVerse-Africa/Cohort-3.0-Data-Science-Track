"""
Machine Learning Evaluation Metrics Implementation

This module provides comprehensive evaluation metrics for both regression and classification tasks.
These metrics are essential for assessing model performance and comparing different algorithms.

The module includes:
- Regression Metrics: MAE, MSE, RMSE, R²
- Classification Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Understanding these metrics is crucial for:
- Model selection and hyperparameter tuning
- Performance comparison between different algorithms
- Identifying model strengths and weaknesses
- Making informed decisions about model deployment
"""

import numpy as np


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE measures the average magnitude of errors in predictions, without considering
    their direction. It's the average of absolute differences between predicted and
    actual values.
    
    Mathematical Formula:
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Properties:
    - Range: [0, +∞)
    - MAE = 0: perfect predictions
    - Higher MAE: worse performance
    - Less sensitive to outliers than MSE/RMSE
    - Same units as the target variable
    
    Use Cases:
    - When you want to understand the average error magnitude
    - When outliers should not have disproportionate influence
    - When interpretability is important (same units as target)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated target values
        
    Returns:
    --------
    mae : float
        Mean absolute error value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).
    
    MSE measures the average of squared differences between predicted and actual values.
    It penalizes larger errors more heavily than smaller ones due to the squaring operation.
    
    Mathematical Formula:
    MSE = (1/n) * Σ(y_true - y_pred)²
    
    Properties:
    - Range: [0, +∞)
    - MSE = 0: perfect predictions
    - Higher MSE: worse performance
    - More sensitive to outliers than MAE
    - Units are squared compared to target variable
    
    Use Cases:
    - When large errors are particularly undesirable
    - For optimization (differentiable, convex)
    - When you want to penalize outliers heavily
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated target values
        
    Returns:
    --------
    mse : float
        Mean squared error value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred)**2)


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE is the square root of MSE, which brings the units back to the same scale
    as the target variable while maintaining the property of penalizing larger errors.
    
    Mathematical Formula:
    RMSE = √MSE = √[(1/n) * Σ(y_true - y_pred)²]
    
    Properties:
    - Range: [0, +∞)
    - RMSE = 0: perfect predictions
    - Higher RMSE: worse performance
    - More sensitive to outliers than MAE
    - Same units as the target variable
    - Always >= MAE (by Jensen's inequality)
    
    Use Cases:
    - When you want MSE's sensitivity to outliers but with interpretable units
    - For model comparison and selection
    - When large errors are particularly undesirable
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated target values
        
    Returns:
    --------
    rmse : float
        Root mean squared error value
    """
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    """
    Calculate R-squared (Coefficient of Determination).
    
    R² measures the proportion of variance in the dependent variable that is
    predictable from the independent variables. It indicates how well the model
    explains the variability in the data.
    
    Mathematical Formula:
    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = Σ(y_true - y_pred)² (sum of squares of residuals)
    - SS_tot = Σ(y_true - ȳ)² (total sum of squares)
    - ȳ = mean of y_true
    
    Properties:
    - Range: (-∞, 1]
    - R² = 1: perfect predictions (model explains all variance)
    - R² = 0: model performs as well as predicting the mean
    - R² < 0: model performs worse than predicting the mean
    - Higher R²: better performance
    
    Interpretation:
    - R² = 0.8 means the model explains 80% of the variance
    - R² = 0 means the model is no better than predicting the mean
    - R² < 0 means the model is worse than a horizontal line at the mean
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated target values
        
    Returns:
    --------
    r2 : float
        R-squared value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Calculate sum of squares of residuals
    ss_res = np.sum((y_true - y_pred)**2)
    
    # Calculate total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    # Calculate R², handling division by zero
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for binary classification.
    
    A confusion matrix is a table used to evaluate the performance of a classification
    model. It shows the counts of true positives, false positives, true negatives,
    and false negatives.
    
    Matrix Layout:
    [[TN, FP],
     [FN, TP]]
    
    Where:
    - TN (True Negative): Correctly predicted negative class
    - FP (False Positive): Incorrectly predicted positive class (Type I error)
    - FN (False Negative): Incorrectly predicted negative class (Type II error)
    - TP (True Positive): Correctly predicted positive class
    
    Use Cases:
    - Understanding model performance in detail
    - Calculating other metrics (precision, recall, F1)
    - Identifying which types of errors the model makes
    - Visualizing classification performance
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
        
    Returns:
    --------
    confusion_matrix : array-like, shape (2, 2)
        Confusion matrix with layout [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    # Count each type of prediction
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred):
    """
    Calculate accuracy score.
    
    Accuracy is the proportion of correct predictions among the total number of cases.
    It's the most intuitive performance measure.
    
    Mathematical Formula:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Properties:
    - Range: [0, 1]
    - Accuracy = 1: perfect predictions
    - Accuracy = 0: all predictions wrong
    - Higher accuracy: better performance
    
    Limitations:
    - Can be misleading with imbalanced datasets
    - Doesn't distinguish between types of errors
    - May not be the best metric for skewed classes
    
    Use Cases:
    - When classes are balanced
    - As a general performance indicator
    - When all types of errors are equally important
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
        
    Returns:
    --------
    accuracy : float
        Accuracy score between 0 and 1
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """
    Calculate precision score.
    
    Precision measures the proportion of positive predictions that are actually correct.
    It answers the question: "Of all positive predictions, how many were correct?"
    
    Mathematical Formula:
    Precision = TP / (TP + FP)
    
    Properties:
    - Range: [0, 1]
    - Precision = 1: no false positives
    - Precision = 0: all positive predictions are wrong
    - Higher precision: fewer false positives
    
    Interpretation:
    - High precision: model is conservative in predicting positive class
    - Low precision: model has many false positives
    
    Use Cases:
    - When false positives are costly (e.g., spam detection)
    - When you want to minimize Type I errors
    - In information retrieval (precision@k)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
        
    Returns:
    --------
    precision : float
        Precision score between 0 and 1
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]  # True Positives
    fp = cm[0, 1]  # False Positives
    
    # Handle division by zero
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """
    Calculate recall (sensitivity) score.
    
    Recall measures the proportion of actual positives that are correctly identified.
    It answers the question: "Of all actual positives, how many did we catch?"
    
    Mathematical Formula:
    Recall = TP / (TP + FN)
    
    Properties:
    - Range: [0, 1]
    - Recall = 1: no false negatives
    - Recall = 0: no true positives found
    - Higher recall: fewer false negatives
    
    Interpretation:
    - High recall: model finds most positive cases
    - Low recall: model misses many positive cases
    
    Use Cases:
    - When false negatives are costly (e.g., medical diagnosis)
    - When you want to minimize Type II errors
    - In information retrieval (recall@k)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
        
    Returns:
    --------
    recall : float
        Recall score between 0 and 1
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]  # True Positives
    fn = cm[1, 0]  # False Negatives
    
    # Handle division by zero
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred):
    """
    Calculate F1 score.
    
    F1 score is the harmonic mean of precision and recall. It provides a single
    metric that balances both precision and recall, giving equal weight to both.
    
    Mathematical Formula:
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Properties:
    - Range: [0, 1]
    - F1 = 1: perfect precision and recall
    - F1 = 0: either precision or recall is 0
    - Higher F1: better balance between precision and recall
    
    Why Harmonic Mean:
    - Harmonic mean is more sensitive to low values than arithmetic mean
    - Ensures that both precision and recall must be high for good F1 score
    - Penalizes models that are good at only one metric
    
    Use Cases:
    - When you need a single metric to optimize
    - When both precision and recall are important
    - With imbalanced datasets (better than accuracy)
    - For model comparison and selection
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
        
    Returns:
    --------
    f1 : float
        F1 score between 0 and 1
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    # Handle division by zero
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0