"""CatBoost model diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve, 
    auc, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from sklearn.inspection import permutation_importance
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_catboost_plots(model=None, X_train=None, y_train=None, X_test=None, y_test=None,
                         feature_names=None, class_names=None, is_classifier=True,
                         learning_rates=None, tree_depths=None, sample_weights=None):
    """Generate diagnostic plots for CatBoost models.
    
    Args:
        model: Fitted CatBoost model
        X_train: Training feature matrix
        y_train: Training target variable
        X_test: Test feature matrix
        y_test: Test target variable
        feature_names: Names of features (optional)
        class_names: Names of classes for classification (optional)
        is_classifier: Whether model is a classifier (True) or regressor (False)
        learning_rates: List of learning rates if hyperparameter tuning plots are needed
        tree_depths: List of tree depths if hyperparameter tuning plots are needed
        sample_weights: Sample weights for weighted analysis
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if we have necessary data
    if model is None:
        return plots
    
    # Ensure feature names are available or create generic ones
    if X_train is not None and feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]
    
    # Convert pandas DataFrames to numpy arrays if needed
    if X_train is not None and hasattr(X_train, 'values'):
        X_train = X_train.values
    if X_test is not None and hasattr(X_test, 'values'):
        X_test = X_test.values
    if y_train is not None and hasattr(y_train, 'values'):
        y_train = y_train.values
    if y_test is not None and hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Plot 1: Feature Importance
    try:
        plt.figure(figsize=(10, 6))
        
        # Get feature importance
        try:
            if hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # Calculate permutation importance if built-in not available
                if X_test is not None and y_test is not None:
                    result = permutation_importance(model, X_test, y_test, 
                                                  n_repeats=10, random_state=42)
                    importances = result.importances_mean
                else:
                    raise ValueError("Cannot calculate feature importance")
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names if feature_names else [f'Feature {i}' for i in range(len(importances))],
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create plot
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('CatBoost Feature Importance')
            plt.tight_layout()
            
            plots.append({
                "title": "Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the relative importance of each feature in the model. Features with higher importance scores have more influence on the model's predictions. CatBoost's feature importance represents how much each feature contributes to reducing the loss function across all trees."
            })
        except Exception as e:
            print(f"Feature importance plotting error: {e}")
            
    except Exception as e:
        print(f"Error in feature importance plot: {e}")
    
    # Plot 2: Training History (if available)
    try:
        if hasattr(model, 'get_evals_result') and model.get_evals_result():
            plt.figure(figsize=(12, 6))
            
            # Get evaluation results
            evals_result = model.get_evals_result()
            
            # Plot metrics
            for dataset in evals_result:
                for metric_name in evals_result[dataset]:
                    plt.plot(evals_result[dataset][metric_name], 
                           label=f'{dataset}-{metric_name}')
            
            plt.xlabel('Iterations')
            plt.ylabel('Metric Value')
            plt.title('CatBoost Learning Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Training History",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how the model's performance evolves during training across iterations. This helps identify potential overfitting (when validation score worsens while training score improves) and determine the optimal number of iterations."
            })
    except Exception as e:
        print(f"Error in training history plot: {e}")
    
    # Plot 3: Classifier-specific plots
    if is_classifier and X_test is not None and y_test is not None:
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_prob = None
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            
            if class_names is None:
                if len(np.unique(y_test)) == 2:
                    class_names = ['Negative', 'Positive']
                else:
                    class_names = [str(i) for i in range(len(np.unique(y_test)))]
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            plots.append({
                "title": "Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the count of true positive, false positive, true negative, and false negative predictions. The diagonal represents correct predictions, while off-diagonal elements are errors. This helps assess the types of misclassifications the model makes."
            })
            
            # ROC Curve (for binary classification)
            if y_prob is not None and len(np.unique(y_test)) == 2:
                plt.figure(figsize=(8, 6))
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                
                # Plot
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (area = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "ROC Curve",
                    "img_data": get_base64_plot(),
                    "interpretation": "Displays the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various thresholds. The area under the curve (AUC) quantifies model performance, with values closer to 1 indicating better performance."
                })
                
                # Precision-Recall Curve
                plt.figure(figsize=(8, 6))
                
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                pr_auc = auc(recall, precision)
                
                # Plot
                plt.plot(recall, precision, color='green', lw=2,
                       label=f'PR curve (area = {pr_auc:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "Precision-Recall Curve",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the trade-off between precision (positive predictive value) and recall (sensitivity) at different classification thresholds. This is particularly useful for imbalanced datasets where ROC curves might be overly optimistic."
                })
        except Exception as e:
            print(f"Error in classification plots: {e}")
    
    # Plot 4: Regressor-specific plots
    elif not is_classifier and X_test is not None and y_test is not None:
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Residual Plot
            plt.figure(figsize=(10, 6))
            residuals = y_test - y_pred
            
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Plot",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the difference between actual and predicted values. Ideally, residuals should be randomly distributed around zero. Patterns in residuals can indicate model deficiencies like non-linearity or heteroscedasticity."
            })
            
            # Actual vs Predicted Plot
            plt.figure(figsize=(10, 6))
            
            # Plot actual vs predicted values
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            plt.figtext(0.15, 0.8, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            plots.append({
                "title": "Actual vs Predicted Values",
                "img_data": get_base64_plot(),
                "interpretation": f"Compares actual values to model predictions. Points closer to the diagonal line indicate more accurate predictions. The plot includes key regression metrics: RMSE (Root Mean Squared Error: {rmse:.4f}), MAE (Mean Absolute Error: {mae:.4f}), and R² (Coefficient of Determination: {r2:.4f})."
            })
            
            # Residual Distribution Plot
            plt.figure(figsize=(10, 6))
            
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of prediction errors (residuals). Ideally, residuals should follow a normal distribution centered at zero, which would indicate that the model's errors are random and not systematic."
            })
        except Exception as e:
            print(f"Error in regression plots: {e}")
    
    # Plot 5: Prediction Distribution (for classification)
    if is_classifier and X_test is not None and y_test is not None and y_prob is not None:
        try:
            plt.figure(figsize=(10, 6))
            
            if len(np.unique(y_test)) == 2:
                # For binary classification
                pos_probs = y_prob[:, 1]
                
                sns.histplot(pos_probs[y_test == 0], color='red', alpha=0.5, label='Actual: Negative', kde=True)
                sns.histplot(pos_probs[y_test == 1], color='blue', alpha=0.5, label='Actual: Positive', kde=True)
                
                plt.xlabel('Predicted Probability (Positive Class)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Predicted Probabilities')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                # For multi-class, we'll use a heatmap of predicted probabilities
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Get unique classes
                unique_classes = np.sort(np.unique(y_test))
                n_classes = len(unique_classes)
                
                # Create a matrix of average predicted probabilities per class
                prob_matrix = np.zeros((n_classes, n_classes))
                
                for i, true_class in enumerate(unique_classes):
                    true_idx = np.where(y_test == true_class)[0]
                    avg_probs = np.mean(y_prob[true_idx], axis=0)
                    prob_matrix[i] = avg_probs
                
                # Plot heatmap
                sns.heatmap(prob_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                         xticklabels=class_names if class_names else unique_classes,
                         yticklabels=class_names if class_names else unique_classes)
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.title('Average Predicted Probabilities by Class')
            
            plots.append({
                "title": "Prediction Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how confident the model is in its predictions across different classes. For binary classification, it shows the distribution of predicted probabilities separated by the actual class. Ideally, the distributions should be well-separated, indicating the model can distinguish between classes effectively."
            })
        except Exception as e:
            print(f"Error in prediction distribution plot: {e}")
    
    # Plot 6: Tree Visualization (if possible)
    try:
        if hasattr(model, 'plot_tree') and hasattr(model, 'tree_count_'):
            tree_to_plot = min(0, model.tree_count_ - 1)  # Just plot the first tree
            plt.figure(figsize=(15, 10))
            
            model.plot_tree(tree_index=tree_to_plot, figsize=(15, 10))
            plt.title(f'CatBoost Tree #{tree_to_plot}')
            
            plots.append({
                "title": "Tree Visualization",
                "img_data": get_base64_plot(),
                "interpretation": "Visualizes a single decision tree from the CatBoost ensemble. Each node represents a splitting rule, and the paths show how predictions are made. This provides insight into how the model makes decisions but represents only one tree from the entire ensemble."
            })
    except Exception as e:
        # This is optional, so we'll just pass if it fails
        pass
    
    return plots 