"""XGBoost model diagnostic plots."""
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

def generate_xgboost_plots(model=None, X_train=None, y_train=None, X_test=None, y_test=None,
                         feature_names=None, class_names=None, is_classifier=True,
                         learning_rates=None, tree_depths=None, sample_weights=None):
    """Generate diagnostic plots for XGBoost models.
    
    Args:
        model: Fitted XGBoost model
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
        
        # Try different ways to get feature importance based on the model type
        try:
            # For scikit-learn API
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            # For native XGBoost models
            elif hasattr(model, 'get_score'):
                importance_dict = model.get_score(importance_type='gain')
                # Convert to array and handle potential mismatch in feature names
                if feature_names is not None:
                    importances = np.zeros(len(feature_names))
                    for feat, imp in importance_dict.items():
                        if feat in feature_names:
                            importances[feature_names.index(feat)] = imp
                else:
                    importances = np.array(list(importance_dict.values()))
                    feature_names = list(importance_dict.keys())
            else:
                # Calculate permutation importance if built-in not available
                if X_test is not None and y_test is not None:
                    result = permutation_importance(model, X_test, y_test, 
                                                  n_repeats=10, random_state=42)
                    importances = result.importances_mean
                else:
                    raise ValueError("Cannot calculate feature importance without feature_importances_ or test data")
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names if feature_names else [f'Feature {i}' for i in range(len(importances))],
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create plot
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('XGBoost Feature Importance')
            plt.tight_layout()
            
            plots.append({
                "title": "Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the relative importance of each feature in the model. Features with higher importance scores have more influence on the model's predictions. XGBoost calculates feature importance based on how much each feature contributes to reducing the loss function over all trees."
            })
        except Exception as e:
            print(f"Feature importance plotting error: {e}")
            
    except Exception as e:
        print(f"Error in feature importance plot: {e}")
    
    # Plot 2: Training History (if available)
    try:
        if hasattr(model, 'evals_result') and model.evals_result():
            plt.figure(figsize=(12, 6))
            
            # Get evaluation results
            eval_results = model.evals_result()
            
            # Plot each metric
            for dataset in eval_results.keys():
                for metric in eval_results[dataset].keys():
                    plt.plot(eval_results[dataset][metric], 
                           label=f'{dataset}-{metric}')
            
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Metric Score')
            plt.title('XGBoost Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Training History",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how performance metrics evolve during training across boosting iterations. This helps identify potential overfitting (when validation score worsens while training score improves) and determine the optimal number of boosting rounds."
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
                    "interpretation": "Displays the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various thresholds. The area under the curve (AUC) quantifies model performance, with values closer to 1 indicating better performance. A model with no discrimination ability would have an AUC of 0.5 (diagonal line)."
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
                    "interpretation": "Shows the trade-off between precision (positive predictive value) and recall (sensitivity) at different classification thresholds. This is particularly useful for imbalanced datasets where ROC curves might be overly optimistic. The area under the PR curve indicates overall performance, with higher values being better."
                })
            
            # Multi-class ROC Curves
            elif y_prob is not None and len(np.unique(y_test)) > 2:
                plt.figure(figsize=(10, 8))
                
                n_classes = len(np.unique(y_test))
                
                # Binarize the labels for multi-class ROC
                y_test_bin = pd.get_dummies(y_test).values
                
                # Compute ROC curve and ROC area for each class
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr,
                           label=f'Class {class_names[i] if class_names else i} (area = {roc_auc:.3f})')
                
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Multi-class ROC Curves (One-vs-Rest)')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "Multi-class ROC Curves",
                    "img_data": get_base64_plot(),
                    "interpretation": "Displays ROC curves for each class in a one-vs-rest approach. This helps evaluate how well the model discriminates each class from all others. The area under each curve quantifies performance for that specific class."
                })
        
        except Exception as e:
            print(f"Error in classifier plots: {e}")
    
    # Plot 4: Regressor-specific plots
    if not is_classifier and X_test is not None and y_test is not None:
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Actual vs Predicted Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add reference line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.grid(True, alpha=0.3)
            
            # Add metrics to plot
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                   transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plots.append({
                "title": "Actual vs Predicted Values",
                "img_data": get_base64_plot(),
                "interpretation": "Compares the model's predictions against actual values. Points closer to the diagonal red line indicate better predictions. The scatter pattern helps identify regions where the model performs well or poorly. Metrics shown are Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared coefficient (R²)."
            })
            
            # Residual Plot
            plt.figure(figsize=(8, 6))
            residuals = y_test - y_pred
            
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Plot",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the errors (residuals) against predicted values. Ideally, residuals should be randomly distributed around zero (red dashed line) with no pattern. Patterns in residuals can indicate model deficiencies such as missing non-linear relationships or heteroscedasticity."
            })
            
            # Residual Distribution
            plt.figure(figsize=(8, 6))
            
            sns.histplot(residuals, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            
            plots.append({
                "title": "Residual Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of prediction errors (residuals). Ideally, residuals should follow a normal distribution centered at zero (red dashed line). Deviations from normality or shifts away from zero suggest systematic bias in the model's predictions."
            })
        
        except Exception as e:
            print(f"Error in regressor plots: {e}")
    
    # Plot 5: Tree Visualization (for a single tree)
    try:
        # XGBoost has a plot_tree function but we need to save to file and capture
        if hasattr(model, 'get_booster'):
            from xgboost import plot_tree
            
            plt.figure(figsize=(15, 10))
            
            # Plot just the first tree (showing all would be too much)
            plot_tree(model, num_trees=0)
            plt.title('First Tree in XGBoost Model')
            
            plots.append({
                "title": "Tree Visualization",
                "img_data": get_base64_plot(),
                "interpretation": "Visualizes the structure of the first decision tree in the ensemble. Each node shows the split condition, gain in the objective function, and the predicted output value. This helps understand how individual trees make decisions based on feature values."
            })
    except Exception as e:
        # Tree visualization might fail if xgboost plotting utilities aren't available
        pass
    
    # Plot 6: Learning Rate and Tree Depth Analysis
    if learning_rates is not None and tree_depths is not None and X_train is not None and y_train is not None:
        try:
            import xgboost as xgb
            
            # Create grid of parameters
            results = []
            
            for lr in learning_rates:
                for depth in tree_depths:
                    # Create a new model with the given parameters
                    if is_classifier:
                        model_tmp = xgb.XGBClassifier(
                            learning_rate=lr,
                            max_depth=depth,
                            n_estimators=100,
                            random_state=42
                        )
                    else:
                        model_tmp = xgb.XGBRegressor(
                            learning_rate=lr,
                            max_depth=depth,
                            n_estimators=100,
                            random_state=42
                        )
                    
                    # Use cross-validation to evaluate
                    from sklearn.model_selection import cross_val_score
                    
                    scores = cross_val_score(
                        model_tmp, X_train, y_train, 
                        cv=5, 
                        scoring='accuracy' if is_classifier else 'neg_mean_squared_error'
                    )
                    
                    # Record results
                    results.append({
                        'learning_rate': lr,
                        'max_depth': depth,
                        'score': np.mean(scores),
                        'std': np.std(scores)
                    })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            
            # Reshape for heatmap
            heatmap_data = results_df.pivot(
                index='max_depth', 
                columns='learning_rate', 
                values='score'
            )
            
            # Plot
            if is_classifier:
                # For classification, higher accuracy is better
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
            else:
                # For regression, less negative MSE is better
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis_r')
            
            plt.title('XGBoost Hyperparameter Performance')
            plt.xlabel('Learning Rate')
            plt.ylabel('Maximum Tree Depth')
            
            plots.append({
                "title": "Hyperparameter Analysis",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how model performance varies with different combinations of learning rate and maximum tree depth. Lighter colors indicate better performance (higher accuracy for classification or lower error for regression). This plot helps identify optimal hyperparameter settings for the XGBoost model."
            })
        
        except Exception as e:
            print(f"Error in hyperparameter analysis plot: {e}")
    
    # Plot 7: Decision Boundary (for 2D classification problems)
    if is_classifier and X_train is not None and y_train is not None and X_train.shape[1] == 2:
        try:
            plt.figure(figsize=(10, 8))
            
            # Determine plot boundaries
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            
            # Create meshgrid
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, (x_max - x_min) / 100),
                np.arange(y_min, y_max, (y_max - y_min) / 100)
            )
            
            # Get predictions for all grid points
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Create custom colormap
            if len(np.unique(y_train)) <= 10:
                cmap = plt.cm.get_cmap('tab10', len(np.unique(y_train)))
            else:
                cmap = plt.cm.viridis
            
            # Plot decision boundary
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
            
            # Plot training points
            scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                               edgecolors='k', cmap=cmap)
            
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            
            # Add feature names as axis labels if available
            if feature_names and len(feature_names) >= 2:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            else:
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
            
            plt.title('XGBoost Decision Boundary')
            
            # Add legend
            if class_names:
                plt.legend(handles=scatter.legend_elements()[0], 
                         labels=class_names,
                         title="Classes")
            
            plots.append({
                "title": "Decision Boundary",
                "img_data": get_base64_plot(),
                "interpretation": "Visualizes how the XGBoost model partitions the feature space for classification. Background colors represent predicted classes in different regions. Points show the training data, colored by their true class. This plot helps understand how the model makes decisions based on feature values and where decision boundaries between classes lie."
            })
        
        except Exception as e:
            print(f"Error in decision boundary plot: {e}")
    
    # Plot 8: Learning Curve (train/test scores vs training size)
    if X_train is not None and y_train is not None:
        try:
            from sklearn.model_selection import learning_curve
            
            plt.figure(figsize=(10, 6))
            
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy' if is_classifier else 'neg_mean_squared_error',
                n_jobs=-1, random_state=42
            )
            
            # Calculate statistics
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
            plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
            
            # Add bands for standard deviation
            plt.fill_between(train_sizes, train_mean - train_std, 
                           train_mean + train_std, alpha=0.1, color='blue')
            plt.fill_between(train_sizes, test_mean - test_std, 
                           test_mean + test_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Score')
            plt.title('XGBoost Learning Curve')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Learning Curve",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how model performance changes with increasing training data size. The gap between training (blue) and cross-validation (red) scores indicates overfitting. If both curves plateau, more data might not improve performance. If they haven't converged, more training data could be beneficial."
            })
        
        except Exception as e:
            print(f"Error in learning curve plot: {e}")
    
    return plots 