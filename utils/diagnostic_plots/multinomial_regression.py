"""Multinomial regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io
import base64
from itertools import cycle

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_multinomial_regression_plots(X, y):
    """Generate diagnostic plots for multinomial regression
    
    Args:
        X: Features (numpy array)
        y: Multiclass target variable (numpy array)
        
    Returns:
        List of dictionaries with plot information
    """
    # Fit the model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    
    plots = []
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plots.append({
        "title": "Confusion Matrix",
        "img_data": get_base64_plot(),
        "interpretation": "Displays the number of correct and incorrect predictions for each class. The diagonal shows correct predictions, while off-diagonal elements represent misclassifications. Higher values on the diagonal indicate better model performance."
    })
    
    # Plot 2: Class Probability Distribution
    plt.figure(figsize=(10, 6))
    n_classes = len(np.unique(y))
    
    for i, class_idx in enumerate(np.unique(y)):
        class_mask = (y == class_idx)
        for j in range(n_classes):
            plt.hist(y_pred_proba[class_mask, j], alpha=0.5, bins=20, 
                     label=f'True class {class_idx} - Prob of class {j}')
            if i == 0:  # Only add once per target class
                break
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Predicted Probability Distribution by Class')
    
    plots.append({
        "title": "Probability Distribution",
        "img_data": get_base64_plot(),
        "interpretation": "Shows how the model assigns probabilities to each class. Well-separated distributions indicate good discrimination between classes. Look for distributions where the model assigns high probabilities to the correct class."
    })
    
    # Plot 3: ROC Curve (One-vs-Rest)
    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan'])
    
    for i, color in zip(range(n_classes), colors):
        # Convert to binary classification problem (one-vs-rest)
        y_binary = (y == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curves')
    plt.legend(loc="lower right")
    
    plots.append({
        "title": "ROC Curves (One-vs-Rest)",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the model's ability to discriminate each class from the rest at different threshold settings. The AUC (Area Under Curve) indicates performance for each class, with higher AUC values (closer to 1.0) indicating better discrimination."
    })
    
    # Plot 4: Prediction Error Visualization
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot of the first two features
    if X.shape[1] >= 2:
        for i, class_idx in enumerate(np.unique(y)):
            class_mask = (y == class_idx)
            correct_pred = (y_pred == y) & class_mask
            incorrect_pred = (y_pred != y) & class_mask
            
            plt.scatter(X[correct_pred, 0], X[correct_pred, 1], 
                       marker='o', alpha=0.6, label=f'Class {class_idx} (correct)')
            if np.any(incorrect_pred):
                plt.scatter(X[incorrect_pred, 0], X[incorrect_pred, 1], 
                           marker='x', alpha=0.8, s=80, label=f'Class {class_idx} (error)')
    
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Classification Errors in Feature Space')
        plt.legend()
        
        plots.append({
            "title": "Error Visualization",
            "img_data": get_base64_plot(),
            "interpretation": "Visualizes correct and incorrect predictions in the feature space (using the first two features). Misclassified points (marked with 'x') may indicate areas where the model struggles, possibly due to class overlap or complex decision boundaries."
        })
    
    return plots 