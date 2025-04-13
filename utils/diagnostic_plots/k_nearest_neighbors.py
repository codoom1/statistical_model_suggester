"""K-Nearest Neighbors diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_knn_plots(model=None, X_train=None, y_train=None, X_test=None, y_test=None,
                     feature_names=None, class_names=None, is_classifier=True,
                     k_values=None, distance_metrics=None):
    """Generate diagnostic plots for K-Nearest Neighbors models.
    
    Args:
        model: Fitted K-Nearest Neighbors model
        X_train: Training feature matrix
        y_train: Training target variable
        X_test: Test feature matrix
        y_test: Test target variable
        feature_names: Names of features (optional)
        class_names: Names of classes for classification (optional)
        is_classifier: Whether model is a classifier (True) or regressor (False)
        k_values: List of k values for hyperparameter tuning (optional)
        distance_metrics: List of distance metrics for comparison (optional)
        
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
    
    # Plot 1: Decision Boundaries (for 2D classification problems)
    if is_classifier and X_train is not None and y_train is not None and X_train.shape[1] == 2:
        try:
            plt.figure(figsize=(10, 8))
            
            # Create color maps
            n_classes = len(np.unique(y_train))
            if n_classes > 10:
                cmap_light = plt.cm.viridis
                cmap_bold = plt.cm.viridis
            else:
                cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', 
                                          '#AAFFFF', '#EEEEEE', '#FFBB99', '#99FFBB', '#BB99FF'])
                cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                                          '#00FFFF', '#000000', '#FF7700', '#00FF77', '#7700FF'])
            
            # Create mesh grid
            h = 0.02  # step size in the mesh
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Predict class labels for the mesh grid points
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot the decision boundary
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.2)
            
            # Plot the training points
            scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                               edgecolors='k', s=40, cmap=cmap_bold)
            
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            
            # Add feature names if available
            if feature_names and len(feature_names) >= 2:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            else:
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                
            plt.title(f'KNN Decision Boundaries (k={model.n_neighbors})')
            
            # Add legend with class names if available
            if class_names:
                plt.legend(handles=scatter.legend_elements()[0], 
                         labels=class_names, 
                         title="Classes")
            
            plots.append({
                "title": "Decision Boundaries",
                "img_data": get_base64_plot(),
                "interpretation": f"Visualizes how the K-Nearest Neighbors (k={model.n_neighbors}) algorithm partitions the feature space. Colored regions represent different predicted classes, and points show the training data. The boundaries are smoother than decision trees, as KNN makes predictions based on the majority class of the nearest neighbors."
            })
        except Exception as e:
            print(f"Error in decision boundaries plot: {e}")
    
    # Plot 2: Effect of k on Classification Decision Boundaries
    if is_classifier and X_train is not None and y_train is not None and X_train.shape[1] == 2 and k_values is not None:
        try:
            from sklearn.neighbors import KNeighborsClassifier
            
            # Set up figure with subplots
            n_plots = min(len(k_values), 6)  # Limit to 6 plots
            rows = (n_plots + 1) // 2
            cols = min(2, n_plots)
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            if n_plots == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Create color maps
            n_classes = len(np.unique(y_train))
            if n_classes > 10:
                cmap_light = plt.cm.viridis
                cmap_bold = plt.cm.viridis
            else:
                cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', 
                                          '#AAFFFF', '#EEEEEE', '#FFBB99', '#99FFBB', '#BB99FF'])
                cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                                          '#00FFFF', '#000000', '#FF7700', '#00FF77', '#7700FF'])
            
            # Create mesh grid
            h = 0.05  # Coarser step size for faster plotting
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Plot for each k value
            for i, k in enumerate(k_values[:n_plots]):
                # Create and fit model with current k
                knn = KNeighborsClassifier(n_neighbors=k, weights=model.weights,
                                         algorithm=model.algorithm,
                                         p=model.p)
                knn.fit(X_train, y_train)
                
                # Predict class labels for the mesh grid points
                Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot the decision boundary
                axes[i].pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.2)
                
                # Plot the training points
                scatter = axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                                     edgecolors='k', s=20, cmap=cmap_bold)
                
                axes[i].set_xlim(xx.min(), xx.max())
                axes[i].set_ylim(yy.min(), yy.max())
                
                # Add feature names if available
                if feature_names and len(feature_names) >= 2:
                    axes[i].set_xlabel(feature_names[0])
                    axes[i].set_ylabel(feature_names[1])
                else:
                    axes[i].set_xlabel('Feature 1')
                    axes[i].set_ylabel('Feature 2')
                    
                axes[i].set_title(f'KNN (k={k})')
                
                # Add legend with class names if available
                if class_names and i == 0:  # Only add legend to first plot to avoid clutter
                    axes[i].legend(handles=scatter.legend_elements()[0], 
                               labels=class_names,
                               title="Classes")
            
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            
            plots.append({
                "title": "Effect of k on Decision Boundaries",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how different values of k (number of nearest neighbors) affect the decision boundaries. Smaller k values result in more complex boundaries that might overfit, while larger values create smoother boundaries that may underfit. This helps understand the trade-off between model complexity and generalization."
            })
        except Exception as e:
            print(f"Error in k-effect plot: {e}")
    
    # Plot 3: k-Value Performance Curve
    if X_train is not None and y_train is not None and k_values is not None:
        try:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.model_selection import cross_val_score
            
            plt.figure(figsize=(10, 6))
            
            # Store cross-validation scores for each k
            train_scores = []
            cv_scores = []
            cv_std = []
            
            for k in k_values:
                # Create model with current k
                if is_classifier:
                    knn = KNeighborsClassifier(n_neighbors=k, weights=model.weights,
                                           algorithm=model.algorithm, p=model.p)
                    scoring = 'accuracy'
                else:
                    knn = KNeighborsRegressor(n_neighbors=k, weights=model.weights,
                                           algorithm=model.algorithm, p=model.p)
                    scoring = 'neg_mean_squared_error'
                
                # Fit model
                knn.fit(X_train, y_train)
                
                # Calculate training score
                train_scores.append(knn.score(X_train, y_train))
                
                # Calculate cross-validation score
                cv_score = cross_val_score(knn, X_train, y_train, cv=5, scoring=scoring)
                cv_scores.append(np.mean(cv_score))
                cv_std.append(np.std(cv_score))
            
            # Plot train and cross-validation scores
            plt.plot(k_values, train_scores, 'o-', label='Training score')
            plt.plot(k_values, cv_scores, 'o-', label='Cross-validation score')
            
            # Add error bands for cross-validation
            plt.fill_between(k_values, 
                          np.array(cv_scores) - np.array(cv_std),
                          np.array(cv_scores) + np.array(cv_std), 
                          alpha=0.2)
            
            # Find best k value
            best_k_idx = np.argmax(cv_scores) if is_classifier else np.argmin(-np.array(cv_scores))
            best_k = k_values[best_k_idx]
            
            # Highlight best k
            plt.axvline(x=best_k, color='r', linestyle='--')
            plt.text(best_k+0.1, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                   f'Best k = {best_k}', color='r')
            
            plt.xlabel('Number of Neighbors (k)')
            plt.ylabel('Score')
            plt.title('KNN Performance vs Number of Neighbors')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Use log scale if k values span multiple orders of magnitude
            if k_values[-1] / k_values[0] > 100:
                plt.xscale('log')
            
            plots.append({
                "title": "k-Value Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows how model performance changes with different k values. The optimal k value based on cross-validation is {best_k}. As k increases, the model becomes smoother but potentially less accurate for local patterns. The gap between training and cross-validation scores indicates the degree of overfitting."
            })
        except Exception as e:
            print(f"Error in k-value performance plot: {e}")
    
    # Plot 4: Confusion Matrix (for classification)
    if is_classifier and X_test is not None and y_test is not None:
        try:
            plt.figure(figsize=(8, 6))
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Determine class names if not provided
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
            
            # Add accuracy to the plot
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            plt.annotate(f'Accuracy: {accuracy:.3f}', xy=(0.5, 0), xytext=(0.5, -0.1),
                        xycoords='axes fraction', textcoords='axes fraction',
                        ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            
            plots.append({
                "title": "Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the counts of true positives, false positives, true negatives, and false negatives. The diagonal elements represent correct predictions. The overall accuracy is {accuracy:.3f}. This visualization helps identify which classes the KNN model struggles to distinguish correctly."
            })
        except Exception as e:
            print(f"Error in confusion matrix plot: {e}")
    
    # Plot 5: ROC Curve (for binary classification)
    if is_classifier and X_test is not None and y_test is not None and hasattr(model, 'predict_proba'):
        try:
            # Check if binary classification
            if len(np.unique(y_test)) == 2:
                plt.figure(figsize=(8, 6))
                
                # Get probability predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
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
                    "interpretation": f"Shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various probability thresholds. The area under the ROC curve (AUC) of {roc_auc:.3f} quantifies the model's ability to distinguish between classes. Higher AUC values indicate better performance."
                })
            elif len(np.unique(y_test)) > 2:
                # Multiclass ROC curve - one vs rest
                plt.figure(figsize=(10, 8))
                
                # Binarize the output for one-vs-rest ROC
                y_test_bin = pd.get_dummies(y_test).values
                y_pred_proba = model.predict_proba(X_test)
                
                for i in range(len(np.unique(y_train))):
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    class_label = class_names[i] if class_names else f'Class {i}'
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, lw=2,
                           label=f'{class_label} (area = {roc_auc:.3f})')
                
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Multiclass ROC Curves (One-vs-Rest)')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "Multiclass ROC Curves",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows ROC curves for each class in a one-vs-rest approach. Each curve represents how well the model distinguishes between one class and all others. The area under each curve quantifies performance for that specific class."
                })
        except Exception as e:
            print(f"Error in ROC curve plot: {e}")
    
    # Plot 6: Feature Space Visualization with PCA
    if X_train is not None and y_train is not None and X_train.shape[1] > 2:
        try:
            # Apply PCA to reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train)
            
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot with PCA-transformed data
            if is_classifier:
                scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                                   c=y_train, edgecolors='k', alpha=0.7)
                
                # Add legend with class names if available
                if class_names:
                    plt.legend(handles=scatter.legend_elements()[0], 
                             labels=class_names, 
                             title="Classes")
            else:
                scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                                   c=y_train, cmap='viridis', edgecolors='k', alpha=0.7)
                plt.colorbar(label='Target Value')
            
            # Add explained variance information
            explained_var = pca.explained_variance_ratio_
            plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
            
            plt.title('PCA Visualization of Feature Space')
            plt.grid(True, alpha=0.3)
            
            # Indicate k circles for a few example points
            if X_test is not None and len(X_test) > 0:
                # Transform a test point to PCA space
                X_test_pca = pca.transform(X_test)
                
                # Display first test point and its k nearest neighbors
                test_point = X_test_pca[0]
                
                # Draw the test point
                plt.scatter(test_point[0], test_point[1], c='red', 
                          s=100, edgecolors='k', marker='*', label='Test Point')
                
                # Find k nearest neighbors in PCA space
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=model.n_neighbors).fit(X_train_pca)
                distances, indices = nbrs.kneighbors([test_point])
                
                # Draw the neighbors
                plt.scatter(X_train_pca[indices[0], 0], X_train_pca[indices[0], 1], 
                          c='yellow', s=80, edgecolors='k', marker='o', label=f'{model.n_neighbors} Nearest Neighbors')
                
                # Draw circle enclosing the neighbors
                circle_radius = np.max(distances)
                circle = plt.Circle((test_point[0], test_point[1]), circle_radius, 
                                  fill=False, edgecolor='r', linestyle='--')
                plt.gca().add_patch(circle)
                
                plt.legend()
            
            plots.append({
                "title": "Feature Space Visualization",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the data projected onto the first two principal components, which explain {explained_var[0]:.2%} and {explained_var[1]:.2%} of the variance respectively. Each point is a training example, colored by its class or target value. For high-dimensional data, this helps visualize how the KNN algorithm uses feature proximity for predictions."
            })
        except Exception as e:
            print(f"Error in PCA visualization plot: {e}")
    
    # Plot 7: Distance Metrics Comparison
    if is_classifier and X_train is not None and y_train is not None and distance_metrics is not None:
        try:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import cross_val_score
            
            plt.figure(figsize=(12, 6))
            
            # Convert distance metrics to p values for Minkowski
            p_values = []
            metric_labels = []
            
            for metric in distance_metrics:
                if metric == 'euclidean':
                    p_values.append(2)
                    metric_labels.append('Euclidean (p=2)')
                elif metric == 'manhattan':
                    p_values.append(1)
                    metric_labels.append('Manhattan (p=1)')
                elif metric == 'chebyshev':
                    p_values.append(float('inf'))
                    metric_labels.append('Chebyshev (p=inf)')
                elif isinstance(metric, (int, float)):
                    p_values.append(metric)
                    metric_labels.append(f'Minkowski (p={metric})')
                else:
                    p_values.append(None)
                    metric_labels.append(metric)
            
            # Calculate cross-validation scores for each distance metric
            cv_scores = []
            
            for i, metric in enumerate(distance_metrics):
                # Handle Minkowski with different p-values
                if p_values[i] is not None:
                    knn = KNeighborsClassifier(n_neighbors=model.n_neighbors, 
                                           weights=model.weights,
                                           p=p_values[i])
                else:
                    knn = KNeighborsClassifier(n_neighbors=model.n_neighbors, 
                                           weights=model.weights,
                                           metric=metric)
                
                # Calculate cross-validation score
                cv_score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
                cv_scores.append(cv_score)
            
            # Create box plot of scores
            plt.boxplot(cv_scores, labels=metric_labels)
            plt.ylabel('Accuracy')
            plt.title('Performance by Distance Metric')
            plt.grid(True, alpha=0.3)
            
            # Add individual points
            for i, scores in enumerate(cv_scores):
                x = np.random.normal(i+1, 0.04, size=len(scores))
                plt.scatter(x, scores, alpha=0.6)
            
            plt.axhline(y=np.mean(cross_val_score(model, X_train, y_train, cv=5)), 
                       color='r', linestyle='--', 
                       label=f'Current model ({model.metric if hasattr(model, "metric") else "default"})')
            plt.legend()
            
            plt.tight_layout()
            
            plots.append({
                "title": "Distance Metrics Comparison",
                "img_data": get_base64_plot(),
                "interpretation": "Compares model performance with different distance metrics. The choice of distance metric can significantly impact KNN performance depending on the data. Euclidean distance (p=2) works well for continuous variables with similar scales, Manhattan distance (p=1) is less sensitive to outliers, and Chebyshev distance (p=inf) considers only the maximum difference across dimensions."
            })
        except Exception as e:
            print(f"Error in distance metrics comparison plot: {e}")
    
    # Plot 8: Regression-specific plots
    if not is_classifier and X_test is not None and y_test is not None:
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Create figure with 2 subplots
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Actual vs Predicted
            ax[0].scatter(y_test, y_pred, alpha=0.6)
            
            # Add reference line
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax[0].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax[0].set_xlabel('Actual Values')
            ax[0].set_ylabel('Predicted Values')
            ax[0].set_title('Actual vs Predicted Values')
            ax[0].grid(True, alpha=0.3)
            
            # Add metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            ax[0].text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}',
                     transform=ax[0].transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 2: Residuals
            residuals = y_test - y_pred
            ax[1].scatter(y_pred, residuals, alpha=0.6)
            ax[1].axhline(y=0, color='r', linestyle='--')
            
            ax[1].set_xlabel('Predicted Values')
            ax[1].set_ylabel('Residuals')
            ax[1].set_title('Residual Plot')
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plots.append({
                "title": "Regression Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Left: Compares actual values to predicted values. Points close to the red diagonal line indicate accurate predictions. The model achieves RMSE={rmse:.3f}, MAE={mae:.3f}, and R²={r2:.3f}.\n\nRight: Shows residuals (errors) against predicted values. Ideally, residuals should be randomly distributed around the horizontal red line (zero error). Patterns in residuals can indicate model limitations."
            })
        except Exception as e:
            print(f"Error in regression plots: {e}")
            
    # Plot 9: Weighted KNN Assessment
    if 'weights' in model.get_params() and X_train is not None and y_train is not None:
        try:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.model_selection import cross_val_score
            
            plt.figure(figsize=(10, 6))
            
            # Get current k value
            k = model.n_neighbors
            
            # Range of k values to test
            if k_values is None:
                # Create a range around the current k value
                k_range = np.unique([max(1, k-5), max(1, k-2), k, k+2, k+5, k+10])
                k_range = k_range[k_range > 0]  # Ensure all values are positive
            else:
                k_range = k_values
            
            # Array to store scores
            uniform_scores = []
            distance_scores = []
            
            for curr_k in k_range:
                # Create models with uniform and distance weighting
                if is_classifier:
                    uniform_knn = KNeighborsClassifier(n_neighbors=curr_k, weights='uniform',
                                                    algorithm=model.algorithm, p=model.p)
                    distance_knn = KNeighborsClassifier(n_neighbors=curr_k, weights='distance',
                                                     algorithm=model.algorithm, p=model.p)
                    scoring = 'accuracy'
                else:
                    uniform_knn = KNeighborsRegressor(n_neighbors=curr_k, weights='uniform',
                                                   algorithm=model.algorithm, p=model.p)
                    distance_knn = KNeighborsRegressor(n_neighbors=curr_k, weights='distance',
                                                    algorithm=model.algorithm, p=model.p)
                    scoring = 'neg_mean_squared_error'
                
                # Calculate cross-validation scores
                uniform_cv = cross_val_score(uniform_knn, X_train, y_train, cv=5, scoring=scoring)
                distance_cv = cross_val_score(distance_knn, X_train, y_train, cv=5, scoring=scoring)
                
                uniform_scores.append(np.mean(uniform_cv))
                distance_scores.append(np.mean(distance_cv))
            
            # Plot scores
            plt.plot(k_range, uniform_scores, 'o-', label='Uniform weights')
            plt.plot(k_range, distance_scores, 'o-', label='Distance weights')
            
            plt.xlabel('Number of Neighbors (k)')
            plt.ylabel('Score')
            plt.title('Uniform vs Distance-Weighted KNN Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Use log scale if k values span multiple orders of magnitude
            if k_range[-1] / k_range[0] > 100:
                plt.xscale('log')
            
            # Highlight the current model's setting
            current_weights = model.weights
            plt.axvline(x=k, color='r', linestyle='--')
            plt.text(k+0.1, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                   f'Current: k={k}, weights={current_weights}', color='r')
            
            plots.append({
                "title": "Weights Comparison",
                "img_data": get_base64_plot(),
                "interpretation": f"Compares 'uniform' weights (where all neighbors have equal influence) to 'distance' weights (where closer neighbors have more influence). Distance weighting often performs better when k is larger, as it reduces the impact of distant, potentially less relevant neighbors. The current model uses {current_weights} weights with k={k}."
            })
        except Exception as e:
            print(f"Error in weights comparison plot: {e}")
    
    return plots 