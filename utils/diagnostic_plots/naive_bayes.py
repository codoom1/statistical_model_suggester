"""Naive Bayes diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_naive_bayes_plots(model, X=None, y=None, X_test=None, y_test=None, 
                              feature_names=None, class_names=None):
    """Generate diagnostic plots for Naive Bayes models
    
    Args:
        model: Fitted Naive Bayes model (GaussianNB, MultinomialNB, etc.)
        X: Feature matrix
        y: Target variable
        X_test: Test data features (optional)
        y_test: Test data target (optional)
        feature_names: Names of features (optional)
        class_names: Names of classes (optional)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check for required data
    if X is None or y is None:
        return plots
    
    # Get model type
    model_type = model.__class__.__name__
    
    # Get feature names if not provided
    if feature_names is None and X is not None:
        if hasattr(X, 'columns'):  # If X is a DataFrame
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
    
    # Get class names if not provided
    if class_names is None:
        if hasattr(model, 'classes_'):
            class_names = [str(c) for c in model.classes_]
        else:
            unique_classes = np.unique(y)
            class_names = [str(c) for c in unique_classes]
    
    # Plot 1: Confusion Matrix
    if hasattr(model, 'predict'):
        plt.figure(figsize=(10, 8))
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Training Data)')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of correct and incorrect predictions for each class in the training data. The diagonal represents correct predictions, while off-diagonal elements show misclassifications."
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
                      xticklabels=class_names, yticklabels=class_names)
            
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix (Test Data)')
            plt.tight_layout()
            
            plots.append({
                "title": "Test Set Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the count of correct and incorrect predictions for each class in the test data. Compare with the training confusion matrix to assess whether the model generalizes well."
            })
    
    # Plot 2: Feature Distributions by Class (for GaussianNB)
    if model_type == 'GaussianNB' and hasattr(model, 'theta_') and hasattr(model, 'sigma_'):
        # Convert X to numpy array if it's not already
        X_np = np.array(X)
        
        # Plot feature distributions for each feature (up to 6 features)
        n_features = min(6, X_np.shape[1])
        n_classes = len(class_names)
        
        # Create a grid of plots
        fig = plt.figure(figsize=(15, n_features * 3))
        gs = GridSpec(n_features, 1)
        
        for i in range(n_features):
            ax = fig.add_subplot(gs[i])
            
            # Get feature name
            feature_name = feature_names[i] if feature_names else f"Feature {i+1}"
            
            # Calculate feature range
            feature_min = X_np[:, i].min()
            feature_max = X_np[:, i].max()
            x_range = np.linspace(feature_min - 0.1 * (feature_max - feature_min), 
                                feature_max + 0.1 * (feature_max - feature_min), 1000)
            
            # Plot Gaussian distributions for each class
            for j in range(n_classes):
                mean = model.theta_[j, i]
                var = model.sigma_[j, i]
                
                if var > 0:  # Avoid division by zero
                    pdf = np.exp(-(x_range - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
                    ax.plot(x_range, pdf, label=f"Class {class_names[j]}")
                
                # Add vertical line for mean
                ax.axvline(x=mean, color=f'C{j}', linestyle='--', alpha=0.6)
            
            # Plot actual data distribution as a histogram (optional, can be commented out if too cluttered)
            sns.histplot(x=X_np[:, i], hue=y, element="step", stat="density", alpha=0.3, ax=ax)
            
            ax.set_title(f"Feature Distribution: {feature_name}")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Density")
            ax.legend()
        
        plt.tight_layout()
        
        plots.append({
            "title": "Feature Distributions by Class",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the Gaussian probability distributions learned by the model for each feature and class. The dashed vertical lines represent the mean values for each class. This plot helps visualize how the model distinguishes between classes based on feature values."
        })
    
    # Plot 3: Feature Importance (using permutation importance)
    if hasattr(model, 'predict') and feature_names is not None:
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
                "interpretation": "Shows the decrease in model performance when a feature is randomly shuffled. Higher values indicate more important features. This helps identify which features contribute most to the model's predictions."
            })
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
    
    # Plot 4: Class Probability Distribution
    if hasattr(model, 'predict_proba'):
        plt.figure(figsize=(12, 8))
        
        # Get probability predictions
        y_prob = model.predict_proba(X)
        
        # For each class, plot distribution of probabilities
        for i in range(len(class_names)):
            # Get probabilities for current class
            probs = y_prob[:, i]
            
            # Plot KDE for actual positive and negative samples separately
            class_mask = (y == i)
            if np.sum(class_mask) > 0:  # Only plot if class has samples
                sns.kdeplot(probs[class_mask], 
                          label=f"True {class_names[i]}", 
                          shade=True, alpha=0.5)
                
                if np.sum(~class_mask) > 0:  # If there are negative samples
                    sns.kdeplot(probs[~class_mask], 
                              label=f"Not {class_names[i]}", 
                              shade=True, alpha=0.5)
        
        plt.title('Class Probability Distributions')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Class Probability Distributions",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how the model assigns probabilities to each class. Ideally, samples of a given class should receive high probability for that class (curve shifted to the right) and low probability for other classes (curve shifted to the left)."
        })

    # Plot 5: ROC Curve (for binary or multi-class)
    if hasattr(model, 'predict_proba'):
        # Get probability predictions
        y_prob = model.predict_proba(X)
        
        # Get number of classes
        n_classes = len(class_names)
        
        if n_classes == 2:
            # Binary classification case
            plt.figure(figsize=(8, 8))
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
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
                "interpretation": f"Shows the trade-off between true positive rate and false positive rate at different classification thresholds. The area under the curve (AUC) of {roc_auc:.3f} measures the model's ability to discriminate between classes."
            })
            
            # Test data ROC if available
            if X_test is not None and y_test is not None:
                plt.figure(figsize=(8, 8))
                
                # Get probability predictions for test data
                y_test_prob = model.predict_proba(X_test)
                
                # Calculate ROC curve
                fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob[:, 1])
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
                    "interpretation": f"Shows the ROC curve for the test data with AUC of {roc_auc_test:.3f}. Compare with the training ROC curve to check for overfitting."
                })
        else:
            # Multiclass case: one-vs-rest ROC curves
            plt.figure(figsize=(10, 8))
            
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Binarize the labels for one-vs-rest ROC
            y_bin = label_binarize(y, classes=range(n_classes))
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Plot ROC curve for this class
                plt.plot(fpr[i], tpr[i], lw=2, 
                       label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            # Plot diagonal line
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
                "interpretation": "Shows the ROC curve for each class in a one-vs-rest approach. Each curve represents how well the model distinguishes between one class and all others."
            })
    
    # Plot 6: Decision Boundary Visualization (2D only)
    if X.shape[1] >= 2 and hasattr(model, 'predict'):
        # Try with original features if we have 2
        if X.shape[1] == 2:
            features_2d = X
            feature_idx = [0, 1]
            feature_names_2d = [feature_names[0], feature_names[1]]
        else:
            # Use PCA to get 2D representation
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(X)
            feature_idx = None
            feature_names_2d = ['Principal Component 1', 'Principal Component 2']
        
        plt.figure(figsize=(10, 8))
        
        # Define mesh grid for contour plot
        h = 0.02  # step size in the mesh
        x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
        y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Get predictions on the mesh grid
        if feature_idx is not None:
            # Use original features
            mesh_features = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(mesh_features)
        else:
            # Need to transform mesh through PCA inverse transform
            # This is an approximation
            mesh_features = np.c_[xx.ravel(), yy.ravel()]
            
            # Use a simple model to predict the mesh points
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(features_2d, y)
            Z = knn.predict(mesh_features)
        
        # Plot the decision boundary
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
        
        # Plot the training points
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y, 
                           edgecolors='k', alpha=0.6, cmap=plt.cm.Paired)
        
        plt.xlabel(feature_names_2d[0])
        plt.ylabel(feature_names_2d[1])
        plt.title('Decision Boundary Visualization')
        
        # Add a colorbar legend
        if len(class_names) <= 10:  # Only for a reasonable number of classes
            plt.colorbar(scatter, ticks=range(len(class_names)))
            plt.clim(-0.5, len(class_names) - 0.5)
        
        plt.tight_layout()
        
        if feature_idx is not None:
            interpretation = f"Visualizes the decision boundaries of the model in the original feature space using {feature_names[0]} and {feature_names[1]}. Different colors represent different predicted classes."
        else:
            interpretation = "Visualizes the decision boundaries of the model in a reduced 2D space using PCA. Different colors represent different predicted classes. Note that this is an approximation of the actual decision boundaries in the original feature space."
        
        plots.append({
            "title": "Decision Boundary Visualization",
            "img_data": get_base64_plot(),
            "interpretation": interpretation
        })
    
    # Plot 7: Log Probabilities for MultinomialNB (for text classification)
    if model_type == 'MultinomialNB' and hasattr(model, 'feature_log_prob_') and feature_names is not None:
        # Get the log probabilities
        log_probs = model.feature_log_prob_
        
        # Select top features for each class
        n_classes = len(class_names)
        n_top_features = min(10, len(feature_names))
        
        for class_idx in range(n_classes):
            plt.figure(figsize=(12, 8))
            
            # Get top features with highest log probability for this class
            top_indices = np.argsort(-log_probs[class_idx])[:n_top_features]
            top_features = [feature_names[i] for i in top_indices]
            top_log_probs = log_probs[class_idx, top_indices]
            
            # Create bar chart
            y_pos = np.arange(len(top_features))
            plt.barh(y_pos, top_log_probs, align='center')
            plt.yticks(y_pos, top_features)
            plt.xlabel('Log Probability')
            plt.title(f'Top Features for Class: {class_names[class_idx]}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plots.append({
                "title": f"Top Features for Class: {class_names[class_idx]}",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the most indicative features (e.g., words) for class {class_names[class_idx]} in a MultinomialNB model. Higher log probability values indicate features that strongly suggest this class when present."
            })
    
    # Plot 8: Prior Probabilities
    if hasattr(model, 'class_prior_'):
        plt.figure(figsize=(10, 6))
        
        # Plot class priors
        plt.bar(class_names, model.class_prior_)
        plt.ylabel('Prior Probability')
        plt.xlabel('Class')
        plt.title('Class Prior Probabilities')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plots.append({
            "title": "Class Prior Probabilities",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the prior probabilities of each class in the training data. These prior probabilities represent the model's initial belief about the class distribution before considering any features."
        })
    
    return plots 