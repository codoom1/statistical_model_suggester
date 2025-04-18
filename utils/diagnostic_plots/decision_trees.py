"""Decision Trees diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.tree import plot_tree
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

def generate_decision_tree_plots(model=None, X_train=None, y_train=None, X_test=None, y_test=None,
                               feature_names=None, class_names=None, is_classifier=True,
                               max_depths=None, min_samples_splits=None, min_samples_leafs=None):
    """Generate diagnostic plots for Decision Tree models.
    
    Args:
        model: Fitted Decision Tree model
        X_train: Training feature matrix
        y_train: Training target variable
        X_test: Test feature matrix
        y_test: Test target variable
        feature_names: Names of features (optional)
        class_names: Names of classes for classification (optional)
        is_classifier: Whether model is a classifier (True) or regressor (False)
        max_depths: List of max_depth values for hyperparameter tuning
        min_samples_splits: List of min_samples_split values for hyperparameter tuning
        min_samples_leafs: List of min_samples_leaf values for hyperparameter tuning
        
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
    
    # Plot 1: Tree Visualization
    try:
        plt.figure(figsize=(20, 10))
        
        # Use sklearn's plot_tree
        plot_tree(model, 
                filled=True, 
                feature_names=feature_names, 
                class_names=class_names if is_classifier else None,
                rounded=True, 
                fontsize=10,
                proportion=True)
        
        plt.title(f"Decision Tree Visualization", fontsize=14)
        plt.tight_layout()
        
        plots.append({
            "title": "Decision Tree Visualization",
            "img_data": get_base64_plot(),
            "interpretation": "A visualization of the decision tree structure. Each node displays the splitting condition, the impurity measure (Gini index or mean squared error), the number of samples, and the class distribution or mean value at that node. Leaf nodes (in different colors) represent the final decisions or predictions made by the tree."
        })
    except Exception as e:
        print(f"Error in tree visualization: {e}")
    
    # Plot 2: Feature Importance
    try:
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names if feature_names else [f'Feature {i}' for i in range(len(model.feature_importances_))],
                'Importance': model.feature_importances_
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create barplot
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance')
            plt.tight_layout()
            
            plots.append({
                "title": "Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the relative importance of each feature in the decision tree. Features with higher importance scores have more influence on the model's predictions. Importance is calculated based on how much each feature reduces the impurity when used for splitting."
            })
    except Exception as e:
        print(f"Error in feature importance plot: {e}")
    
    # Plot 3: Decision Surface (for 2D classification problems)
    if is_classifier and X_train is not None and y_train is not None and X_train.shape[1] == 2:
        try:
            plt.figure(figsize=(10, 8))
            
            # Set up the meshgrid
            h = 0.02  # step size in the mesh
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Predict on the meshgrid
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundaries
            n_classes = len(np.unique(y_train))
            if n_classes <= 10:
                cmap = plt.cm.tab10
            else:
                cmap = plt.cm.viridis
                
            plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
            
            # Plot training points
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', cmap=cmap)
            
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            
            # Feature labels
            if feature_names and len(feature_names) >= 2:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            else:
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                
            plt.title('Decision Tree Decision Boundaries')
            
            plots.append({
                "title": "Decision Boundaries",
                "img_data": get_base64_plot(),
                "interpretation": "Visualizes how the decision tree partitions the feature space for classification. Different colors represent different predicted classes. The rectangular decision boundaries are characteristic of decision trees, which make splits parallel to the feature axes. This plot helps understand how the model uses the two features to make predictions."
            })
        except Exception as e:
            print(f"Error in decision surface plot: {e}")
    
    # Plot 4: Confusion Matrix (for classification)
    if is_classifier and X_test is not None and y_test is not None:
        try:
            plt.figure(figsize=(8, 6))
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Determine class names
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
            accuracy = accuracy_score(y_test, y_pred)
            plt.annotate(f'Accuracy: {accuracy:.3f}', xy=(0.5, 0), xytext=(0.5, -0.1),
                        xycoords='axes fraction', textcoords='axes fraction',
                        ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            
            plots.append({
                "title": "Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows how the model's predictions compare to the actual classes. The diagonal elements represent correct predictions, while off-diagonal elements are misclassifications. The overall accuracy is {accuracy:.3f}. This helps identify which classes the model struggles to distinguish."
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
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                       label='ROC curve (area = %0.3f)' % roc_auc)
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
                    "interpretation": f"Shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various threshold settings. The area under the ROC curve (AUC) of {roc_auc:.3f} quantifies the model's ability to distinguish between classes. Higher AUC values indicate better performance, with 1.0 being perfect classification."
                })
                
                # Add Precision-Recall curve
                plt.figure(figsize=(8, 6))
                
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                
                # Plot precision-recall curve
                plt.plot(recall, precision, color='green', lw=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.grid(True, alpha=0.3)
                
                # Add average precision to the plot
                avg_precision = np.mean(precision)
                plt.annotate(f'Average Precision: {avg_precision:.3f}', xy=(0.5, 0), xytext=(0.5, -0.1),
                            xycoords='axes fraction', textcoords='axes fraction',
                            ha='center', va='center', fontsize=12)
                
                plots.append({
                    "title": "Precision-Recall Curve",
                    "img_data": get_base64_plot(),
                    "interpretation": f"Shows the trade-off between precision (positive predictive value) and recall (sensitivity) at different threshold settings. This is particularly useful for imbalanced datasets where accuracy can be misleading. The average precision score is {avg_precision:.3f}, with higher values indicating better performance."
                })
        except Exception as e:
            print(f"Error in ROC curve plot: {e}")
    
    # Plot 6: Training vs Validation Performance
    if max_depths is not None and X_train is not None and y_train is not None:
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            
            plt.figure(figsize=(10, 6))
            
            # Scores for different max_depth values
            train_scores = []
            cv_scores = []
            
            for depth in max_depths:
                # Create model with specified depth
                if is_classifier:
                    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
                    scoring = 'accuracy'
                else:
                    tree_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
                    scoring = 'neg_mean_squared_error'
                
                # Fit model
                tree_model.fit(X_train, y_train)
                
                # Get training score
                train_score = tree_model.score(X_train, y_train)
                train_scores.append(train_score)
                
                # Get cross-validation score
                cv_score = np.mean(cross_val_score(tree_model, X_train, y_train, cv=5, scoring=scoring))
                cv_scores.append(cv_score)
            
            # Plot scores
            plt.plot(max_depths, train_scores, 'o-', color='blue', label='Training score')
            plt.plot(max_depths, cv_scores, 'o-', color='red', label='Cross-validation score')
            
            plt.xlabel('Maximum Tree Depth')
            plt.ylabel('Score')
            plt.title('Decision Tree Performance vs Tree Depth')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Mark the optimal depth
            optimal_depth = max_depths[np.argmax(cv_scores)]
            plt.axvline(x=optimal_depth, color='green', linestyle='--')
            plt.text(optimal_depth+0.2, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
                   f'Optimal depth: {optimal_depth}', fontsize=10)
            
            plots.append({
                "title": "Tree Depth Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows how model performance changes with tree depth. The blue line represents training performance and the red line cross-validation performance. As depth increases, the model becomes more complex, potentially leading to overfitting (high training score but decreasing validation score). The optimal depth based on cross-validation is {optimal_depth}."
            })
        except Exception as e:
            print(f"Error in tree depth performance plot: {e}")
    
    # Plot 7: Cost-Complexity Pruning (a.k.a. Alpha tuning)
    if X_train is not None and y_train is not None:
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.model_selection import train_test_split
            
            # Split training data for validation
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, random_state=42)
            
            # Fit a decision tree with default parameters
            if is_classifier:
                tree_model = DecisionTreeClassifier(random_state=42)
            else:
                tree_model = DecisionTreeRegressor(random_state=42)
                
            path = tree_model.cost_complexity_pruning_path(X_tr, y_tr)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            
            # We want to evaluate a reasonable number of alphas, not too many
            if len(ccp_alphas) > 20:
                indices = np.linspace(0, len(ccp_alphas)-1, 20, dtype=int)
                ccp_alphas = ccp_alphas[indices]
                impurities = impurities[indices]
            
            # Train trees with different alphas and evaluate
            train_scores = []
            val_scores = []
            tree_depths = []
            node_counts = []
            
            for alpha in ccp_alphas:
                # Skip very small alphas which might cause numerical issues
                if alpha < 1e-6:
                    continue
                    
                # Create model with specified alpha
                if is_classifier:
                    tree_model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
                else:
                    tree_model = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
                
                # Fit model
                tree_model.fit(X_tr, y_tr)
                
                # Collect metrics
                train_scores.append(tree_model.score(X_tr, y_tr))
                val_scores.append(tree_model.score(X_val, y_val))
                tree_depths.append(tree_model.get_depth())
                node_counts.append(tree_model.get_n_leaves())
            
            # Make sure we have data to plot
            if len(train_scores) > 1:
                # Create figure with 3 subplots
                fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
                
                # Plot 1: Training vs Validation Accuracy
                ax[0].plot(ccp_alphas[1:], train_scores, 'o-', label='Train')
                ax[0].plot(ccp_alphas[1:], val_scores, 'o-', label='Validation')
                ax[0].set_ylabel('Accuracy' if is_classifier else 'R²')
                ax[0].set_title('Performance vs Alpha')
                ax[0].legend()
                ax[0].grid(True, alpha=0.3)
                
                # Plot 2: Tree Depth
                ax[1].plot(ccp_alphas[1:], tree_depths, 'o-')
                ax[1].set_ylabel('Tree Depth')
                ax[1].set_title('Tree Depth vs Alpha')
                ax[1].grid(True, alpha=0.3)
                
                # Plot 3: Number of Leaves
                ax[2].plot(ccp_alphas[1:], node_counts, 'o-')
                ax[2].set_xlabel('Alpha')
                ax[2].set_ylabel('Number of Leaves')
                ax[2].set_title('Number of Leaves vs Alpha')
                ax[2].grid(True, alpha=0.3)
                
                # Use log scale for x-axis if the alphas span multiple orders of magnitude
                alpha_range = max(ccp_alphas[1:]) / min(ccp_alphas[1:])
                if alpha_range > 100:
                    for a in ax:
                        a.set_xscale('log')
                
                plt.tight_layout()
                
                plots.append({
                    "title": "Cost-Complexity Pruning",
                    "img_data": get_base64_plot(),
                    "interpretation": "These plots show the effects of cost-complexity pruning (controlled by alpha parameter). As alpha increases, the tree is pruned more aggressively, resulting in simpler trees with fewer nodes. The top plot shows how performance changes with different alpha values, helping identify the optimal pruning level that balances model complexity and performance."
                })
        except Exception as e:
            print(f"Error in cost-complexity pruning plot: {e}")
    
    # Plot 8: Predictor Space Partitioning (for 2D problems)
    if X_train is not None and X_train.shape[1] == 2:
        try:
            plt.figure(figsize=(10, 8))
            
            # Get all the thresholds and features from the tree
            thresholds = []
            features = []
            
            # Recursive function to get all decision boundaries
            def get_tree_boundaries(tree, node_id=0):
                # If not a leaf node, add the boundary
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    thresholds.append(tree.threshold[node_id])
                    features.append(tree.feature[node_id])
                    
                    # Process children recursively
                    get_tree_boundaries(tree, tree.children_left[node_id])
                    get_tree_boundaries(tree, tree.children_right[node_id])
            
            # Get boundaries from the tree
            get_tree_boundaries(model.tree_)
            
            # Determine feature ranges
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            
            # Plot data points
            if is_classifier and y_train is not None:
                plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=15, cmap='viridis', alpha=0.6)
            else:
                plt.scatter(X_train[:, 0], X_train[:, 1], s=15, alpha=0.6)
            
            # Plot decision boundaries
            for i, (feature, threshold) in enumerate(zip(features, thresholds)):
                if feature == 0:  # X-axis feature
                    plt.axvline(x=threshold, color='red', linestyle='-', alpha=0.3)
                elif feature == 1:  # Y-axis feature
                    plt.axhline(y=threshold, color='blue', linestyle='-', alpha=0.3)
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            # Feature labels
            if feature_names and len(feature_names) >= 2:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            else:
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                
            plt.title('Decision Tree Partitioning of Feature Space')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label=f'Split on {feature_names[0] if feature_names else "Feature 1"}'),
                Line2D([0], [0], color='blue', lw=2, label=f'Split on {feature_names[1] if feature_names else "Feature 2"}')
            ]
            plt.legend(handles=legend_elements)
            
            plots.append({
                "title": "Feature Space Partitioning",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how the decision tree partitions the feature space into regions. Red lines represent splits on the first feature (x-axis), while blue lines represent splits on the second feature (y-axis). This visualization illustrates the recursive binary splitting process of decision trees and the resulting rectangular decision regions."
            })
        except Exception as e:
            print(f"Error in predictor space partitioning plot: {e}")
    
    # Plot 9: Regression-specific plots
    if not is_classifier and X_test is not None and y_test is not None:
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Create figure with 2 subplots
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Actual vs Predicted
            ax[0].scatter(y_test, y_pred, alpha=0.6)
            
            # Add reference line
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax[0].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax[0].set_xlabel('Actual Values')
            ax[0].set_ylabel('Predicted Values')
            ax[0].set_title('Actual vs Predicted Values')
            
            # Add performance metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
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
            
            plt.tight_layout()
            
            plots.append({
                "title": "Regression Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Left: Actual vs Predicted values. Points closer to the red diagonal line indicate better predictions. Performance metrics are RMSE (Root Mean Squared Error) of {rmse:.3f}, MAE (Mean Absolute Error) of {mae:.3f}, and R² (coefficient of determination) of {r2:.3f}.\n\nRight: Residual plot showing prediction errors against predicted values. Ideally, residuals should be randomly distributed around zero (red dashed line) with no pattern. Patterns in residuals can indicate model deficiencies."
            })
        except Exception as e:
            print(f"Error in regression plots: {e}")
    
    return plots 