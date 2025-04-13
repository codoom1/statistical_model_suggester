"""Gradient boosting diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_gradient_boosting_plots(model, X=None, y=None, X_test=None, y_test=None, 
                                   feature_names=None, class_names=None, is_classifier=None):
    """Generate diagnostic plots for gradient boosting models
    
    Args:
        model: Fitted gradient boosting model (XGBoost, LightGBM, CatBoost, or sklearn GBM)
        X: Feature matrix
        y: Target variable
        X_test: Test data features (optional)
        y_test: Test data target (optional)
        feature_names: Names of features (optional)
        class_names: Names of classes for classification problems (optional)
        is_classifier: Whether the model is a classifier (if None, will be auto-detected)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Determine if model is a classifier if not specified
    if is_classifier is None:
        # Check model attributes or predict method
        if hasattr(model, 'classes_'):
            is_classifier = True
        elif hasattr(model, 'n_classes_') and model.n_classes_ > 0:
            is_classifier = True
        elif hasattr(model, '_classes'):
            is_classifier = True
        elif hasattr(model, 'objective') and isinstance(model.objective, str):
            is_classifier = 'class' in model.objective or 'binary' in model.objective
        else:
            # Try to infer from y
            if y is not None:
                unique_values = np.unique(y)
                is_classifier = len(unique_values) <= 10 and all(isinstance(val, (int, np.integer, bool)) or val.is_integer() for val in unique_values)
            else:
                # Default to regressor
                is_classifier = False
    
    # Get feature names if not provided
    if feature_names is None and X is not None:
        if hasattr(X, 'columns'):  # If X is a DataFrame
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
    
    # Try to get class names if not provided and is classifier
    if class_names is None and is_classifier:
        if hasattr(model, 'classes_'):
            class_names = [str(c) for c in model.classes_]
        elif y is not None:
            unique_classes = np.unique(y)
            class_names = [str(c) for c in unique_classes]
    
    # Plot 1: Feature importance
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot importances
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        plots.append({
            "title": "Feature Importance",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the relative importance of each feature in the model. Higher values indicate features that have more influence on the model's predictions. The importance metric depends on the specific gradient boosting implementation, but generally represents how useful a feature was for building the boosted trees."
        })
    elif X is not None and y is not None:
        # If model doesn't have feature_importances_ attribute, use permutation importance
        try:
            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
            
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot importances
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Permutation Feature Importance')
            plt.tight_layout()
            
            plots.append({
                "title": "Permutation Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the decrease in model performance when a feature is randomly shuffled. Higher values indicate more important features. Unlike built-in feature importance, permutation importance is not biased toward high-cardinality features and better reflects the actual impact on model predictions."
            })
        except:
            pass
    
    # Plot 2: Learning curve (training vs validation error over iterations)
    if hasattr(model, 'evals_result_'):
        # XGBoost style
        try:
            plt.figure(figsize=(10, 6))
            
            evals_result = model.evals_result()
            for eval_set, metrics in evals_result.items():
                for metric, values in metrics.items():
                    plt.plot(range(len(values)), values, label=f'{eval_set} - {metric}')
            
            plt.xlabel('Boosting Iteration')
            plt.ylabel('Error')
            plt.title('Training and Validation Error per Iteration')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Learning Curve",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how the model's error changes with each boosting iteration on both training and validation data. Increasing validation error while training error continues to decrease indicates overfitting. The optimal number of iterations should minimize validation error."
            })
        except:
            pass
    elif hasattr(model, 'train_score_'):
        # sklearn GradientBoosting style
        plt.figure(figsize=(10, 6))
        
        # Plot training deviance
        plt.plot(np.arange(len(model.train_score_)) + 1, model.train_score_, 'b-',
                label='Training Set Deviance', alpha=0.8)
        
        # Plot test deviance if available
        if hasattr(model, 'validation_score_'):
            plt.plot(np.arange(len(model.validation_score_)) + 1, model.validation_score_, 'r-',
                    label='Validation Set Deviance', alpha=0.8)
        
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        plt.title('Deviance over Boosting Iterations')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Deviance Curve",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how the model's deviance (a measure of error) changes with each boosting iteration. A flattening curve indicates diminishing returns from additional iterations. If validation deviance increases while training deviance continues to decrease, this suggests overfitting."
        })
    
    # Check prediction capabilities
    has_predict = hasattr(model, 'predict')
    has_predict_proba = hasattr(model, 'predict_proba') and is_classifier
    
    # Plot 3: For regression, actual vs predicted values
    if not is_classifier and has_predict and X is not None and y is not None:
        plt.figure(figsize=(10, 8))
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Scatter plot
        plt.scatter(y, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Add metrics annotation
        plt.annotate(f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows how well the model's predictions match actual values. Points closer to the red diagonal line indicate better predictions. RMSE (Root Mean Squared Error) of {rmse:.3f} penalizes large errors, while MAE (Mean Absolute Error) of {mae:.3f} shows average error magnitude. R² of {r2:.3f} indicates the proportion of variance explained (closer to 1 is better)."
        })
        
        # Plot for test data if provided
        if X_test is not None and y_test is not None:
            plt.figure(figsize=(10, 8))
            
            # Get test predictions
            y_test_pred = model.predict(X_test)
            
            # Scatter plot
            plt.scatter(y_test, y_test_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_test_pred))
            max_val = max(max(y_test), max(y_test_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Actual Values (Test Set)')
            plt.ylabel('Predicted Values (Test Set)')
            plt.title('Test Set: Actual vs Predicted Values')
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics for test set
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Add metrics annotation
            plt.annotate(f'RMSE: {test_rmse:.3f}\nMAE: {test_mae:.3f}\nR²: {test_r2:.3f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plots.append({
                "title": "Test Set Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows model performance on unseen test data. Compare these metrics (RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}) with training metrics to assess overfitting. Similar performance suggests good generalization, while significantly worse test performance indicates potential overfitting."
            })
            
            # Plot 4: Residual plot
            plt.figure(figsize=(10, 6))
            
            # Calculate residuals
            residuals = y_test - y_test_pred
            
            # Scatter plot
            plt.scatter(y_test_pred, residuals, alpha=0.6)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color='r', linestyle='--')
            
            # Add trend line
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smooth = lowess(residuals, y_test_pred, frac=0.6)
                plt.plot(smooth[:, 0], smooth[:, 1], 'g-', lw=2)
            except:
                pass
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted Values (Test Set)')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Analysis",
                "img_data": get_base64_plot(),
                "interpretation": "Examines patterns in prediction errors. Ideally, residuals should be randomly scattered around zero (red line) with no clear pattern. Systematic patterns indicate that the model hasn't captured some aspect of the data relationship. Funneling patterns suggest heteroscedasticity (error variance depends on prediction value)."
            })
    
    # Plot 5: For classification, confusion matrix
    if is_classifier and has_predict and X is not None and y is not None:
        plt.figure(figsize=(10, 8))
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else "auto",
                   yticklabels=class_names if class_names else "auto")
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Displays the count of correct and incorrect predictions for each class. The diagonal cells show correct predictions, while off-diagonal cells represent misclassifications. This helps identify which classes are most confused with each other."
        })
        
        # Plot for test data if provided
        if X_test is not None and y_test is not None:
            plt.figure(figsize=(10, 8))
            
            # Get test predictions
            y_test_pred = model.predict(X_test)
            
            # Compute confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names if class_names else "auto",
                       yticklabels=class_names if class_names else "auto")
            
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Test Set: Confusion Matrix')
            plt.tight_layout()
            
            plots.append({
                "title": "Test Set Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the model's classification performance on unseen test data. Compare with the training confusion matrix to check for consistency. Similar performance indicates good generalization ability."
            })
    
    # Plot 6: For binary classification, ROC curve
    if is_classifier and has_predict_proba and X is not None and y is not None:
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            plt.figure(figsize=(8, 8))
            
            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X)[:, 1]
            else:
                y_score = model.decision_function(X)
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
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
                "interpretation": f"Shows the trade-off between true positive rate and false positive rate at different classification thresholds. The area under the curve (AUC) of {roc_auc:.3f} measures the model's ability to discriminate between classes. AUC ranges from 0.5 (random guessing) to 1.0 (perfect classification)."
            })
            
            # Plot Precision-Recall curve
            plt.figure(figsize=(8, 8))
            
            # Compute PR curve and average precision
            precision, recall, _ = precision_recall_curve(y, y_score)
            avg_precision = average_precision_score(y, y_score)
            
            # Plot PR curve
            plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Precision-Recall Curve",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the trade-off between precision and recall at different classification thresholds. The average precision (AP) of {avg_precision:.3f} summarizes the curve. Particularly useful for imbalanced datasets where ROC curve may be overly optimistic."
            })
        elif n_classes > 2:
            # Multiclass ROC curves (one-vs-rest)
            plt.figure(figsize=(10, 8))
            
            # Compute ROC curve and ROC area for each class
            y_binary = np.zeros((len(y), n_classes))
            for i in range(n_classes):
                y_binary[:, i] = (y == i).astype(int)
            
            # Get probability predictions
            y_score = model.predict_proba(X)
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_binary[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                class_label = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
                plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (One-vs-Rest)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Multiclass ROC Curves",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the ROC curve for each class in a one-vs-rest approach. Each curve represents how well the model distinguishes that class from all others. Higher AUC values indicate better class discrimination."
            })
    
    # Plot 7: Calibration plot for classification
    if is_classifier and has_predict_proba and X is not None and y is not None:
        try:
            from sklearn.calibration import calibration_curve
            
            plt.figure(figsize=(10, 8))
            
            # Get probability predictions
            if n_classes == 2:
                # Binary classifier
                prob_pos = model.predict_proba(X)[:, 1]
                
                # Calculate calibration curve
                prob_true, prob_pred = calibration_curve(y, prob_pos, n_bins=10)
                
                # Plot calibration curve
                plt.plot(prob_pred, prob_true, "s-", label=f"Gradient Boosting")
                
                # Plot perfectly calibrated line
                plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                
                plt.xlabel("Mean predicted probability")
                plt.ylabel("Fraction of positives")
                plt.title("Calibration Curve")
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "Probability Calibration",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows how well the predicted probabilities match actual outcomes. A perfectly calibrated model should follow the diagonal line. Points above the line indicate underprediction (actual probability > predicted), while points below indicate overprediction."
                })
            else:
                # For multiclass, show histogram of predicted probabilities for each class
                plt.figure(figsize=(12, 4 * n_classes))
                
                # Get probability predictions
                y_prob = model.predict_proba(X)
                
                # Create subplots for each class
                for i in range(n_classes):
                    plt.subplot(n_classes, 1, i + 1)
                    
                    # Get probabilities for this class
                    class_probs = y_prob[:, i]
                    
                    # Plot histograms
                    plt.hist(class_probs[y == i], bins=20, alpha=0.8, density=True, 
                            label=f'True {class_names[i] if class_names else i}')
                    plt.hist(class_probs[y != i], bins=20, alpha=0.5, density=True, 
                            label=f'False {class_names[i] if class_names else i}')
                    
                    plt.xlabel('Predicted Probability')
                    plt.ylabel('Density')
                    plt.title(f'Probability Distribution for {class_names[i] if class_names else f"Class {i}"}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plots.append({
                    "title": "Class Probability Distributions",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the distribution of predicted probabilities for each class, separated by true class membership. Well-separated distributions indicate good discrimination ability. Ideally, true members of a class should have high predicted probabilities for that class."
                })
        except:
            pass
    
    # Plot 8: Partial dependence plots for top features
    try:
        from sklearn.inspection import plot_partial_dependence
        
        if X is not None and feature_names is not None:
            # Get top features (max 4)
            if hasattr(model, 'feature_importances_'):
                top_features = np.argsort(model.feature_importances_)[-4:]
            else:
                # If no feature_importances_, use first 4 features
                top_features = range(min(4, len(feature_names)))
            
            # Create partial dependence plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, feature_idx in enumerate(top_features):
                if i < len(axes):
                    pdp_display = plot_partial_dependence(
                        model, X, [feature_idx], feature_names=feature_names,
                        ax=axes[i], line_kw={"color": "blue"}
                    )
            
            plt.tight_layout()
            
            plots.append({
                "title": "Partial Dependence Plots",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how each feature affects predictions when all other features are held constant. The y-axis represents the change in the predicted value as the feature value changes. These plots help visualize the relationship between each feature and the target, which may be highly non-linear in gradient boosting models."
            })
    except:
        pass
    
    # Plot 9: Tree structure visualization for a sample tree (if small enough)
    try:
        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            # For sklearn gradient boosting
            from sklearn.tree import plot_tree
            
            # Get a sample tree (the first one)
            sample_tree = model.estimators_[0, 0] if is_classifier else model.estimators_[0]
            
            # Only plot if tree is not too large
            if sample_tree.tree_.node_count < 20:
                plt.figure(figsize=(15, 10))
                
                plot_tree(sample_tree, feature_names=feature_names, filled=True, 
                         class_names=class_names if is_classifier else None)
                
                plt.title('Sample Tree from Ensemble (First Tree)')
                
                plots.append({
                    "title": "Sample Tree Visualization",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the structure of a single decision tree from the ensemble. Each node shows the splitting condition, the impurity measure, sample count, and class distribution (for classification) or predicted value (for regression). This provides insight into how the model makes individual decisions, though the full ensemble combines many such trees."
                })
    except:
        pass
    
    return plots 