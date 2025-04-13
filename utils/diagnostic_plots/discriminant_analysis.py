"""
Discriminant Analysis (LDA/QDA) diagnostic plots.
This module provides functionality to generate diagnostic plots for Linear Discriminant Analysis (LDA)
and Quadratic Discriminant Analysis (QDA) models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import io
import base64
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

def get_base64_plot():
    """Convert a matplotlib figure to a base64 encoded string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def generate_discriminant_analysis_plots(model, X=None, y=None, X_test=None, y_test=None, 
                                        feature_names=None, class_names=None):
    """Generate diagnostic plots for discriminant analysis models (LDA, QDA)
    
    Args:
        model: Fitted discriminant analysis model (LDA or QDA)
        X: Feature matrix (training data)
        y: Target variable (training data)
        X_test: Test data features (optional)
        y_test: Test data target (optional)
        feature_names: Names of features (optional)
        class_names: Names of classes (optional)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if we have necessary data to generate plots
    if X is None or y is None:
        return plots
    
    # If class names not provided, generate default ones
    if class_names is None:
        unique_classes = np.unique(y)
        class_names = [f"Class {i}" for i in unique_classes]
    
    # If feature names not provided, generate default ones
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Determine if model is LDA or QDA
    model_type = "Discriminant Analysis"
    if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
        if 'LinearDiscriminantAnalysis' in model.__class__.__name__:
            model_type = "Linear Discriminant Analysis (LDA)"
        elif 'QuadraticDiscriminantAnalysis' in model.__class__.__name__:
            model_type = "Quadratic Discriminant Analysis (QDA)"
    
    # 1. Decision Boundary Visualization (2D projection if needed)
    plt.figure(figsize=(10, 6))
    
    # If more than 2 dimensions, use PCA to project to 2D
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        feature_names_2d = ['PC1', 'PC2']
        
        # Create mesh grid for decision boundary
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Project the mesh grid points to original feature space
        mesh_pca_space = np.c_[xx.ravel(), yy.ravel()]
        try:
            # Try to inverse transform to get back to original feature space
            mesh_points = pca.inverse_transform(mesh_pca_space)
            
            # Predict class for each point in the mesh
            Z = model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        except:
            # If inverse transform fails, we'll skip the decision boundary
            pass
            
        # Scatter plot of data points
        for i, cls in enumerate(np.unique(y)):
            plt.scatter(X_2d[y == cls, 0], X_2d[y == cls, 1], alpha=0.8, 
                        label=class_names[i], edgecolor='k')
        
        plt.title(f'Decision Boundary ({model_type})')
        plt.xlabel(f'{feature_names_2d[0]} (PCA projection)')
        plt.ylabel(f'{feature_names_2d[1]} (PCA projection)')
        
    else:
        # Create mesh grid for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Predict class for each point in the mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        # Scatter plot of data points
        for i, cls in enumerate(np.unique(y)):
            plt.scatter(X[y == cls, 0], X[y == cls, 1], alpha=0.8, 
                        label=class_names[i], edgecolor='k')
        
        plt.title(f'Decision Boundary ({model_type})')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    
    plt.legend()
    plots.append({
        "title": "Decision Boundary",
        "img_data": get_base64_plot(),
        "interpretation": f"Visualizes the decision boundary created by the {model_type} model. Each color represents a different class region. Points are colored by their true class, so points in the 'wrong' color region represent potential misclassifications."
    })
    
    # 2. Class Means and Covariance Visualization
    if hasattr(model, 'means_'):
        plt.figure(figsize=(10, 6))
        
        # If more than 2 dimensions, use PCA to project to 2D
        if X.shape[1] > 2:
            means_2d = pca.transform(model.means_)
            
            # Scatter plot of class means
            plt.scatter(means_2d[:, 0], means_2d[:, 1], s=200, c='red', 
                        marker='*', edgecolor='k', label='Class Means')
            
            # Add class labels to means
            for i, (x, y) in enumerate(means_2d):
                plt.annotate(class_names[i], (x, y), xytext=(10, 5),
                           textcoords='offset points', ha='center')
            
            # If LDA, plot discriminant directions
            if hasattr(model, 'scalings_') and 'Linear' in model_type:
                # Project discriminant directions
                scalings_2d = pca.transform(model.scalings_[:, :2])
                
                # Plot discriminant directions as arrows from origin
                origin = np.zeros(2)
                for i in range(min(scalings_2d.shape[1], len(class_names) - 1)):
                    plt.arrow(origin[0], origin[1], 
                             scalings_2d[0, i], scalings_2d[1, i],
                             head_width=0.1, head_length=0.2, fc='blue', ec='blue',
                             label=f'Discriminant {i+1}' if i==0 else "")
            
            # Plot data points with reduced opacity
            for i, cls in enumerate(np.unique(y)):
                plt.scatter(X_2d[y == cls, 0], X_2d[y == cls, 1], alpha=0.3)
                
            plt.title(f'Class Means and Covariance (PCA projection)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
        else:
            # Scatter plot of class means
            plt.scatter(model.means_[:, 0], model.means_[:, 1], s=200, c='red', 
                        marker='*', edgecolor='k', label='Class Means')
            
            # Add class labels to means
            for i, (x, y) in enumerate(model.means_):
                plt.annotate(class_names[i], (x, y), xytext=(10, 5),
                           textcoords='offset points', ha='center')
            
            # If LDA, plot discriminant directions
            if hasattr(model, 'scalings_') and 'Linear' in model_type:
                # Plot discriminant directions as arrows from origin
                origin = np.zeros(2)
                for i in range(min(model.scalings_.shape[1], len(class_names) - 1)):
                    plt.arrow(origin[0], origin[1], 
                             model.scalings_[0, i], model.scalings_[1, i],
                             head_width=0.1, head_length=0.2, fc='blue', ec='blue',
                             label=f'Discriminant {i+1}' if i==0 else "")
            
            # For QDA, plot covariance ellipses
            if 'Quadratic' in model_type and hasattr(model, 'covariance_'):
                for i, cov in enumerate(model.covariance_):
                    # Calculate eigenvalues and eigenvectors for the covariance matrix
                    evals, evecs = np.linalg.eigh(cov[:2, :2])
                    # Sort by eigenvalue in decreasing order
                    idx = np.argsort(evals)[::-1]
                    evals = evals[idx]
                    evecs = evecs[:, idx]
                    
                    # Calculate angle and width/height
                    angle = np.arctan2(evecs[1, 0], evecs[0, 0]) * 180 / np.pi
                    width, height = 2 * np.sqrt(evals)
                    
                    # Draw ellipse (95% confidence)
                    ellipse = Ellipse(xy=model.means_[i, :2], width=width*2, height=height*2,
                                    angle=angle, alpha=0.3, color=plt.cm.tab10(i))
                    plt.gca().add_patch(ellipse)
            
            # Plot data points with reduced opacity
            for i, cls in enumerate(np.unique(y)):
                plt.scatter(X[y == cls, 0], X[y == cls, 1], alpha=0.3)
            
            plt.title(f'Class Means and Covariance')
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        
        plt.legend()
        plots.append({
            "title": "Class Means and Covariance",
            "img_data": get_base64_plot(),
            "interpretation": "Displays the centroid (mean) of each class and visualizes their relationship. " +
                             ("For LDA, discriminant axes show the directions of maximum class separation. " if 'Linear' in model_type else "") +
                             ("For QDA, ellipses represent the class-specific covariances (95% confidence regions). " if 'Quadratic' in model_type else "") +
                             "Greater separation between class means typically indicates better classification performance."
        })
    
    # 3. Confusion Matrix
    if y_test is not None and X_test is not None:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Show all ticks and label them with class names
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Display values in the cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of true vs. predicted classes. The diagonal elements represent correctly classified instances, while off-diagonal elements are misclassifications. High values along the diagonal and low values elsewhere indicate good model performance."
        })
    
    # 4. Feature Importance (using permutation importance if test data available)
    if X_test is not None and y_test is not None:
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            
            # Sort features by importance
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance')
            
            plots.append({
                "title": "Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the relative importance of each feature to the model's predictions. Features with higher importance values contribute more to the model's discriminatory power. This is calculated using permutation importance, which measures how model performance decreases when a feature is randomly shuffled."
            })
        except:
            pass
    
    # 5. ROC Curves (for multi-class using one-vs-rest)
    if hasattr(model, 'predict_proba') and X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        y_pred_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # One-vs-Rest approach for multiclass
        for i, class_name in enumerate(class_names):
            # For each class, treat it as positive and all others as negative
            y_test_binary = (y_test == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        
        plots.append({
            "title": "ROC Curves",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the trade-off between true positive rate and false positive rate at various classification thresholds. Each curve represents one class vs. all others. AUC (Area Under Curve) values closer to 1 indicate better classification performance. The diagonal line represents a random classifier (AUC = 0.5)."
        })
    
    # 6. Cross-validation Performance
    try:
        # Calculate cross-validation accuracy
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(cv_scores)), cv_scores)
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='-', label=f'Mean: {cv_scores.mean():.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('5-Fold Cross-Validation Performance')
        plt.xticks(range(len(cv_scores)), [f'Fold {i+1}' for i in range(len(cv_scores))])
        plt.legend()
        
        plots.append({
            "title": "Cross-Validation Performance",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows model accuracy across 5 different data folds. Consistent performance across folds suggests the model is robust. Mean accuracy: {cv_scores.mean():.3f}. High variance across folds may indicate that the model is sensitive to the specific data it's trained on."
        })
    except:
        pass
    
    # 7. Discriminant Score Distributions (if LDA)
    if hasattr(model, 'transform') and 'Linear' in model_type:
        try:
            # Transform the data to discriminant space
            X_lda = model.transform(X)
            
            plt.figure(figsize=(10, 6))
            
            # Plot histogram for each class along the first discriminant
            for i, cls in enumerate(np.unique(y)):
                plt.hist(X_lda[y == cls, 0], alpha=0.5, bins=20, 
                       label=class_names[i])
            
            plt.title('First Discriminant Score Distribution by Class')
            plt.xlabel('Discriminant Score')
            plt.ylabel('Frequency')
            plt.legend()
            
            plots.append({
                "title": "Discriminant Score Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of discriminant scores for each class. Good separation in these distributions indicates that the discriminant function effectively differentiates between classes. Overlapping distributions suggest classes that are difficult to separate."
            })
        except:
            pass
    
    # 8. Classifier Metrics Summary
    if X_test is not None and y_test is not None:
        try:
            # Get classification report
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            
            # Prepare data for visualization
            classes = list(report.keys())[:-3]  # Exclude avg metrics
            precision = [report[c]['precision'] for c in classes]
            recall = [report[c]['recall'] for c in classes]
            f1 = [report[c]['f1-score'] for c in classes]
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            x = np.arange(len(classes))
            width = 0.25
            
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1-score')
            
            plt.xlabel('Class')
            plt.ylabel('Score')
            plt.title('Classification Metrics by Class')
            plt.xticks(x, classes)
            plt.ylim(0, 1.1)
            
            # Add a horizontal line for ideal performance
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(precision):
                plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            for i, v in enumerate(recall):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            for i, v in enumerate(f1):
                plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.legend()
            plt.tight_layout()
            
            plots.append({
                "title": "Classification Metrics Summary",
                "img_data": get_base64_plot(),
                "interpretation": "Summarizes precision, recall, and F1-score for each class. Precision measures the proportion of positive identifications that were actually correct. Recall measures the proportion of actual positives that were identified correctly. F1-score is the harmonic mean of precision and recall. Higher values indicate better performance."
            })
        except:
            pass
    
    return plots 