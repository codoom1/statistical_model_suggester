"""Support Vector Machine diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_svm_plots(model, X=None, y=None, X_test=None, y_test=None, 
                      feature_names=None, class_names=None, is_classifier=None):
    """Generate diagnostic plots for SVM models
    
    Args:
        model: Fitted SVM model
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
        if hasattr(model, 'classes_'):
            is_classifier = True
        elif hasattr(model, 'predict_proba'):
            is_classifier = True
        elif hasattr(model, '_impl') and 'classification' in str(model._impl).lower():
            is_classifier = True
        else:
            # Try to infer from y
            if y is not None:
                unique_values = np.unique(y)
                is_classifier = len(unique_values) <= 10
            else:
                is_classifier = False
    
    # Get feature names if not provided
    if feature_names is None and X is not None:
        if hasattr(X, 'columns'):  # If X is a DataFrame
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
    
    # Plot 1: Support vectors visualization (if 2D or reduced to 2D)
    if X is not None and hasattr(model, 'support_vectors_'):
        # Check if data is already 2D or needs dimensionality reduction
        if X.shape[1] == 2:
            # Data is already 2D
            X_2d = X
            feature_1, feature_2 = 0, 1
            feat1_name = feature_names[0] if feature_names else "Feature 1"
            feat2_name = feature_names[1] if feature_names else "Feature 2"
        else:
            # Reduce to 2D using PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            feature_1, feature_2 = 0, 1
            feat1_name = "PC1"
            feat2_name = "PC2"
        
        plt.figure(figsize=(10, 8))
        
        # Get support vectors
        support_vectors = model.support_vectors_
        
        # If support vectors are not in the original space, transform them
        if X.shape[1] != 2:
            support_vectors_2d = pca.transform(support_vectors)
        else:
            support_vectors_2d = support_vectors
        
        # Plot data points
        if is_classifier:
            # For classification, color by class
            unique_classes = np.unique(y)
            for cls in unique_classes:
                idx = y == cls
                class_label = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
                plt.scatter(X_2d[idx, feature_1], X_2d[idx, feature_2], 
                          alpha=0.6, label=class_label)
        else:
            # For regression, use a single color
            plt.scatter(X_2d[:, feature_1], X_2d[:, feature_2], 
                      alpha=0.6, label="Data Points")
        
        # Plot support vectors
        plt.scatter(support_vectors_2d[:, feature_1], support_vectors_2d[:, feature_2], 
                  s=100, linewidth=1, facecolors='none', edgecolors='k', 
                  label="Support Vectors")
        
        # Add decision boundaries if classifier with 2D original data
        if is_classifier and X.shape[1] == 2:
            # Create a mesh grid
            h = 0.02  # step size in the mesh
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                              np.arange(y_min, y_max, h))
            
            # Get decision function values
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            
            # If multiclass, take the first decision function
            if Z.ndim > 1:
                Z = Z[:, 0]
            
            # Reshape to match the mesh grid
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary (Z=0) and margins (Z=±1)
            plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
                      alpha=0.5, linestyles=['--', '-', '--'])
        
        plt.xlabel(feat1_name)
        plt.ylabel(feat2_name)
        plt.title('Support Vectors Visualization')
        plt.legend()
        plt.tight_layout()
        
        if X.shape[1] == 2:
            interpretation = "Shows the data points and identified support vectors (circled in black) in the original feature space. The support vectors are the critical points that define the decision boundary. For classifiers, the solid line represents the decision boundary, while dashed lines show the margins."
        else:
            interpretation = "Shows the data points and identified support vectors (circled in black) in a reduced 2D space using PCA. The support vectors are the critical points that define the decision boundary in the original feature space."
        
        plots.append({
            "title": "Support Vectors Visualization",
            "img_data": get_base64_plot(),
            "interpretation": interpretation
        })
    
    # Plot 2: Feature importance (using permutation importance)
    if X is not None and y is not None and feature_names is not None:
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
                "title": "Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the decrease in model performance when a feature is randomly shuffled. Higher values indicate more important features. Unlike some other models, SVMs don't have built-in feature importance scores, so this permutation-based approach helps understand which features are most influential."
            })
        except:
            pass
    
    # Plot 3: Confusion matrix for classification
    if is_classifier and X is not None and y is not None:
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
        plt.title('Confusion Matrix (Training Data)')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of correct and incorrect predictions for each class in the training data. The diagonal represents correct predictions, while off-diagonal elements show misclassifications. This helps identify which classes are most confused with each other."
        })
        
        # Test data confusion matrix if available
        if X_test is not None and y_test is not None:
            plt.figure(figsize=(10, 8))
            
            # Get predictions on test data
            y_test_pred = model.predict(X_test)
            
            # Compute confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names if class_names else "auto",
                      yticklabels=class_names if class_names else "auto")
            
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix (Test Data)')
            plt.tight_layout()
            
            plots.append({
                "title": "Test Set Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the count of correct and incorrect predictions for each class in the test data. Compare with the training confusion matrix to assess whether the model generalizes well. Similar patterns suggest good generalization."
            })
    
    # Plot 4: ROC curve for classification if predict_proba is available
    if is_classifier and hasattr(model, 'predict_proba') and X is not None and y is not None:
        # Binary classification
        if len(np.unique(y)) == 2:
            plt.figure(figsize=(8, 8))
            
            # Get probability predictions
            y_score = model.predict_proba(X)[:, 1]
            
            # Compute ROC curve and area
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
                "interpretation": f"Shows the trade-off between true positive rate and false positive rate at different classification thresholds. The area under the curve (AUC) of {roc_auc:.3f} measures the model's ability to discriminate between classes. AUC ranges from 0.5 (random) to 1.0 (perfect)."
            })
            
            # Test data ROC if available
            if X_test is not None and y_test is not None:
                plt.figure(figsize=(8, 8))
                
                # Get probability predictions for test data
                y_test_score = model.predict_proba(X_test)[:, 1]
                
                # Compute ROC curve and area
                fpr_test, tpr_test, _ = roc_curve(y_test, y_test_score)
                roc_auc_test = auc(fpr_test, tpr_test)
                
                # Plot ROC curve
                plt.plot(fpr_test, tpr_test, lw=2, label=f'ROC curve (AUC = {roc_auc_test:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (Test Data)')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "Test Set ROC Curve",
                    "img_data": get_base64_plot(),
                    "interpretation": f"Shows the ROC curve for the test data with AUC of {roc_auc_test:.3f}. Compare with the training ROC curve to check for overfitting. Similar AUC values suggest good generalization."
                })
        else:
            # Multiclass ROC (one-vs-rest)
            plt.figure(figsize=(10, 8))
            
            # Compute ROC curve and ROC area for each class
            n_classes = len(np.unique(y))
            y_score = model.predict_proba(X)
            
            for i in range(n_classes):
                # Convert to binary one-vs-rest encoding
                y_binary = (y == i).astype(int)
                
                # Compute ROC curve and area
                fpr, tpr, _ = roc_curve(y_binary, y_score[:, i])
                roc_auc = auc(fpr, tpr)
                
                # Class label
                class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                
                # Plot ROC curve
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
                "interpretation": "Shows the ROC curve for each class in a one-vs-rest approach. Each curve represents how well the model distinguishes between one class and all others. Higher AUC values indicate better class discrimination."
            })
    
    # Plot 5: For regression, predicted vs actual
    if not is_classifier and X is not None and y is not None:
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
        plt.title('Actual vs Predicted Values (Training Data)')
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
            "interpretation": f"Shows how well the model's predictions match actual values in the training data. Points closer to the red diagonal line indicate better predictions. RMSE of {rmse:.3f} penalizes large errors, MAE of {mae:.3f} shows average error magnitude, and R² of {r2:.3f} indicates the proportion of variance explained."
        })
        
        # Test data predictions if available
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
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values (Test Data)')
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics for test data
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
                "interpretation": f"Shows model performance on unseen test data. Compare these metrics (RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}) with training metrics to assess overfitting. Similar performance suggests good generalization."
            })
            
            # Residual analysis
            plt.figure(figsize=(10, 6))
            
            # Calculate residuals
            residuals = y_test - y_test_pred
            
            # Scatter plot
            plt.scatter(y_test_pred, residuals, alpha=0.6)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted Values (Test Data)')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Analysis",
                "img_data": get_base64_plot(),
                "interpretation": "Examines patterns in prediction errors. Ideally, residuals should be randomly scattered around zero (red line) with no clear pattern. Systematic patterns indicate that the model hasn't captured some aspect of the data relationship."
            })
    
    # Plot 6: Hyperparameter sensitivity (if C attribute is present)
    if hasattr(model, 'C'):
        try:
            from sklearn.svm import SVC, SVR
            from sklearn.model_selection import validation_curve
            
            plt.figure(figsize=(10, 6))
            
            # Define the parameter values to test
            param_range = np.logspace(-3, 3, 7)
            
            # Calculate validation curve
            if X is not None and y is not None:
                if is_classifier:
                    estimator = SVC(kernel=model.kernel, gamma=model.gamma) if hasattr(model, 'kernel') else SVC()
                    scoring = 'accuracy'
                else:
                    estimator = SVR(kernel=model.kernel, gamma=model.gamma) if hasattr(model, 'kernel') else SVR()
                    scoring = 'neg_mean_squared_error'
                
                train_scores, test_scores = validation_curve(
                    estimator, X, y, param_name="C", param_range=param_range,
                    cv=5, scoring=scoring, n_jobs=-1
                )
                
                # Calculate mean and std for training scores
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                
                # Calculate mean and std for test scores
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                
                # Scale scores if using negative metrics
                if scoring.startswith('neg_'):
                    train_scores_mean = -train_scores_mean
                    test_scores_mean = -test_scores_mean
                
                # Plot scores
                plt.plot(param_range, train_scores_mean, 'o-', label="Training score", color="b")
                plt.fill_between(param_range, train_scores_mean - train_scores_std,
                               train_scores_mean + train_scores_std, alpha=0.15, color="b")
                
                plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score", color="g")
                plt.fill_between(param_range, test_scores_mean - test_scores_std,
                               test_scores_mean + test_scores_std, alpha=0.15, color="g")
                
                # Mark current C value
                plt.axvline(x=model.C, color='r', linestyle='--', label=f'Current C={model.C}')
                
                plt.xscale('log')
                plt.xlabel('C parameter')
                plt.ylabel('Score')
                plt.title('Validation Curve for C parameter')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "C Parameter Sensitivity",
                    "img_data": get_base64_plot(),
                    "interpretation": f"Shows how model performance changes with different values of the C parameter. C controls the penalty for misclassification, with smaller values creating a wider margin and larger values focusing on classifying training points correctly. Your model uses C={model.C}."
                })
        except:
            pass
    
    # Plot 7: Kernel comparison (if kernel attribute is present)
    if hasattr(model, 'kernel') and X is not None and y is not None:
        try:
            from sklearn.svm import SVC, SVR
            from sklearn.model_selection import cross_val_score
            
            plt.figure(figsize=(10, 6))
            
            # Define kernels to compare
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            
            # Calculate scores for each kernel
            scores = []
            std_errors = []
            
            for kernel in kernels:
                if is_classifier:
                    estimator = SVC(kernel=kernel, C=model.C, gamma='scale')
                    scoring = 'accuracy'
                else:
                    estimator = SVR(kernel=kernel, C=model.C, gamma='scale')
                    scoring = 'neg_mean_squared_error'
                
                # Perform cross-validation
                cv_scores = cross_val_score(estimator, X, y, cv=5, scoring=scoring)
                
                # Scale scores if using negative metrics
                if scoring.startswith('neg_'):
                    cv_scores = -cv_scores
                
                scores.append(np.mean(cv_scores))
                std_errors.append(np.std(cv_scores))
            
            # Create a bar chart
            x_pos = np.arange(len(kernels))
            plt.bar(x_pos, scores, yerr=std_errors, align='center', alpha=0.7)
            
            # Highlight current kernel
            current_kernel_idx = kernels.index(model.kernel) if model.kernel in kernels else -1
            if current_kernel_idx >= 0:
                plt.bar(current_kernel_idx, scores[current_kernel_idx], color='r', alpha=0.7)
            
            plt.xticks(x_pos, kernels)
            plt.xlabel('Kernel Function')
            plt.ylabel('Performance Score')
            plt.title('Performance Comparison of Different Kernels')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Kernel Comparison",
                "img_data": get_base64_plot(),
                "interpretation": f"Compares the performance of different kernel functions. Your model uses the '{model.kernel}' kernel (shown in red). Different kernels are suitable for different types of data relationships: linear for linearly separable data, rbf for complex non-linear boundaries, poly for polynomial relationships, and sigmoid for certain neural network-like patterns."
            })
        except:
            pass
    
    return plots 