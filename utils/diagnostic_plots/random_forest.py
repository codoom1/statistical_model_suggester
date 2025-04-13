"""Random Forest diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import io
import base64

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_random_forest_plots(X, y, feature_names=None, classification=True):
    """Generate diagnostic plots for Random Forest
    
    Args:
        X: Features (numpy array)
        y: Target variable (numpy array)
        feature_names: List of feature names (optional)
        classification: Whether it's a classification or regression task
        
    Returns:
        List of dictionaries with plot information
    """
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Fit the model
    if classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    
    model.fit(X, y)
    
    plots = []
    
    # Plot 1: Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(min(10, X.shape[1])), importances[indices[:10]])
    plt.xticks(range(min(10, X.shape[1])), [feature_names[i] for i in indices[:10]], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the relative importance of each feature in the model. Higher values indicate features that contribute more to predictions. Important for feature selection and understanding the key drivers in your model."
    })
    
    # Plot 2: Permutation Importance (more reliable than default importance)
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_indices = perm_importance.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(perm_importance.importances[perm_indices[:10]].T, 
                vert=False, labels=[feature_names[i] for i in perm_indices[:10]])
    plt.title('Permutation Importance (Top 10 Features)')
    plt.xlabel('Decrease in Accuracy/R²')
    plt.tight_layout()
    plots.append({
        "title": "Permutation Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Measures importance by randomly shuffling feature values and observing the decrease in model performance. Less prone to bias than standard importance for correlated features. Box plots show variation over multiple permutations."
    })
    
    # Plot 3: OOB Error vs Number of Trees
    n_estimators = np.arange(1, 101)
    oob_errors = []
    
    for n in n_estimators[::10]:  # Sample every 10th value for efficiency
        if classification:
            rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42)
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators[::10], oob_errors, 'o-')
    plt.xlabel('Number of Trees')
    plt.ylabel('Out-of-Bag Error')
    plt.title('OOB Error vs Number of Trees')
    plots.append({
        "title": "OOB Error",
        "img_data": get_base64_plot(),
        "interpretation": "Shows how error rate changes with the number of trees. The curve should plateau, indicating an optimal number of trees. If it continues to decrease, more trees might improve performance."
    })
    
    # Plot 4: Decision Tree Visualization (first tree in the forest)
    from sklearn.tree import plot_tree
    
    plt.figure(figsize=(12, 8))
    first_tree = model.estimators_[0]
    plot_tree(first_tree, 
              feature_names=feature_names, 
              filled=True, 
              max_depth=3,  # Limit depth for visualization
              fontsize=10)
    plt.title('Example Decision Tree (First Tree in Forest, Limited to Depth 3)')
    plots.append({
        "title": "Sample Tree Structure",
        "img_data": get_base64_plot(),
        "interpretation": "Visualizes a single tree from the forest, showing how decisions are made. Each node shows the splitting criterion, samples, and class distribution or mean value. This helps understand the decision-making process."
    })
    
    return plots 