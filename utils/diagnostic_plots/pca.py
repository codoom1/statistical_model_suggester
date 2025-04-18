"""Principal Component Analysis diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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

def generate_pca_plots(X, feature_names=None):
    """Generate diagnostic plots for PCA
    
    Args:
        X: Features (numpy array)
        feature_names: List of feature names (optional)
        
    Returns:
        List of dictionaries with plot information
    """
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Standardize data (important for PCA)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    
    plots = []
    
    # Plot 1: Scree Plot
    plt.figure(figsize=(10, 6))
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
    plt.step(range(1, len(cum_explained_var) + 1), cum_explained_var, where='mid', label='Cumulative')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.title('Scree Plot')
    plt.tight_layout()
    plots.append({
        "title": "Scree Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the proportion of variance explained by each principal component. The cumulative line helps determine how many components to retain. Components explaining little variance (flat part of the curve) can usually be ignored."
    })
    
    # Plot 2: Biplot (first two components)
    plt.figure(figsize=(10, 6))
    # Plot observations
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    
    # Scale loading vectors for visualization
    scaling = np.min([np.abs(X_pca[:, 0].min()), np.abs(X_pca[:, 0].max()),
                      np.abs(X_pca[:, 1].min()), np.abs(X_pca[:, 1].max())])
    
    # Plot feature vectors
    for i, (x, y) in enumerate(zip(pca.components_[0], pca.components_[1])):
        plt.arrow(0, 0, x*scaling, y*scaling, head_width=scaling*0.05, head_length=scaling*0.05, fc='r', ec='r')
        plt.text(x*scaling*1.1, y*scaling*1.1, feature_names[i], color='r')
    
    plt.grid(True)
    plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
    plt.title('PCA Biplot')
    plt.tight_layout()
    plots.append({
        "title": "PCA Biplot",
        "img_data": get_base64_plot(),
        "interpretation": "Visualizes both samples (points) and variables (arrows) in the PCA space. Arrows show how original variables contribute to the principal components. Variables pointing in similar directions are positively correlated, while opposite directions indicate negative correlation."
    })
    
    # Plot 3: Loading Plot for PC1
    plt.figure(figsize=(10, 6))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    plt.bar(range(len(loadings)), loadings[:, 0])
    plt.axhline(y=0, color='gray', linestyle='-')
    plt.xticks(range(len(loadings)), feature_names, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Loading on PC1')
    plt.title('PC1 Loadings')
    plt.tight_layout()
    plots.append({
        "title": "PC1 Loadings",
        "img_data": get_base64_plot(),
        "interpretation": "Shows how strongly each original feature influences the first principal component. Larger absolute values indicate more important features in defining this dimension. Signs indicate the direction of correlation with the component."
    })
    
    # Plot 4: Correlation Circle
    plt.figure(figsize=(10, 10))
    
    # Draw the unit circle
    circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='gray')
    plt.gca().add_patch(circle)
    
    # Plot arrows
    for i, (x, y) in enumerate(zip(pca.components_[0], pca.components_[1])):
        plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='b', ec='b')
        plt.text(x*1.1, y*1.1, feature_names[i], fontsize=10)
    
    # Draw labels and adjust plot
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(True)
    plt.axvline(0, color='gray', linestyle='-')
    plt.axhline(0, color='gray', linestyle='-')
    plt.xlabel(f'PC1 ({explained_var[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_var[1]:.2%})')
    plt.title('Correlation Circle')
    plt.tight_layout()
    plots.append({
        "title": "Correlation Circle",
        "img_data": get_base64_plot(),
        "interpretation": "Shows correlations between variables and principal components. Variables closer to the circle edge and to each other are strongly correlated. Variables at 90° are uncorrelated, while variables at 180° are negatively correlated."
    })
    
    return plots 