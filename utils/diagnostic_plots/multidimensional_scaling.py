"""Multidimensional Scaling (MDS) diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from sklearn.metrics import euclidean_distances
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_mds_plots(model=None, X=None, distances=None, labels=None, feature_names=None, 
                    color_values=None, original_distances=None, n_components=2, 
                    metric=True, compare_methods=False):
    """Generate diagnostic plots for Multidimensional Scaling models.
    
    Args:
        model: Fitted MDS model (optional)
        X: Original data matrix (optional if model is provided)
        distances: Precomputed distance matrix (optional)
        labels: Labels for data points (optional)
        feature_names: Names of features (optional)
        color_values: Values for coloring points (optional)
        original_distances: Original high-dimensional distances (for assessing fit quality)
        n_components: Number of components for new models (default=2)
        metric: Whether to use metric MDS (True) or non-metric MDS (False)
        compare_methods: Whether to compare metric and non-metric MDS
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Get embeddings from model or fit new model
    if model is not None:
        # Extract embeddings from model
        embeddings = model.embedding_
        stress = model.stress_
        n_components = embeddings.shape[1]
    elif X is not None or distances is not None:
        # Compute model
        mds = MDS(n_components=n_components, 
                metric=metric, 
                dissimilarity="precomputed" if distances is not None else "euclidean",
                random_state=42)
        
        if distances is not None:
            embeddings = mds.fit_transform(distances)
        else:
            embeddings = mds.fit_transform(X)
            
        stress = mds.stress_
    else:
        return plots  # No data to plot
    
    # Compute distances if needed for later plots
    if original_distances is None and X is not None:
        original_distances = euclidean_distances(X)
    
    # Plot 1: 2D MDS Embedding
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        
        if color_values is not None:
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                               c=color_values, cmap='viridis', 
                               s=50, alpha=0.8, edgecolors='w')
            plt.colorbar(label='Value')
        else:
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                               s=50, alpha=0.8, edgecolors='w')
        
        # Add labels if provided
        if labels is not None:
            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings[i, 0], embeddings[i, 1]), 
                           fontsize=9, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'MDS Embedding (Stress: {stress:.2f})')
        plt.grid(True, alpha=0.3)
        
        # Add stress value as a text box
        ax = plt.gca()
        plt.text(0.95, 0.05, f'Stress: {stress:.4f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plots.append({
            "title": "2D MDS Embedding",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows data points projected onto the first two dimensions of MDS. Points that are close in this 2D space have similar features in the original high-dimensional space. The stress value of {stress:.4f} indicates how well the distances in the 2D space match the original distances - lower values are better."
        })
    
    # Plot 2: 3D MDS Embedding (if available)
    if n_components >= 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if color_values is not None:
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                                  c=color_values, cmap='viridis', 
                                  s=50, alpha=0.8, edgecolors='w')
                plt.colorbar(scatter, label='Value')
            else:
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                                  s=50, alpha=0.8, edgecolors='w')
            
            # Add labels if provided
            if labels is not None:
                for i, label in enumerate(labels):
                    ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], 
                          label, fontsize=9, alpha=0.8)
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.set_title(f'3D MDS Embedding (Stress: {stress:.2f})')
            
            plt.tight_layout()
            
            plots.append({
                "title": "3D MDS Embedding",
                "img_data": get_base64_plot(),
                "interpretation": "Shows data points projected onto the first three dimensions of MDS. This gives a more complete view of the data structure compared to the 2D plot, potentially revealing patterns that aren't visible in just two dimensions."
            })
        except Exception as e:
            print(f"Error generating 3D plot: {e}")
    
    # Plot 3: Shepard Diagram (original distances vs. MDS distances)
    if original_distances is not None:
        plt.figure(figsize=(8, 8))
        
        # Compute MDS distances
        mds_distances = euclidean_distances(embeddings)
        
        # Flatten the distance matrices for plotting
        original_dist_flat = original_distances.flatten()
        mds_dist_flat = mds_distances.flatten()
        
        # Remove self-distances (zeros)
        mask = original_dist_flat != 0
        original_dist_flat = original_dist_flat[mask]
        mds_dist_flat = mds_dist_flat[mask]
        
        # Calculate correlations
        pearson_corr, _ = pearsonr(original_dist_flat, mds_dist_flat)
        spearman_corr, _ = spearmanr(original_dist_flat, mds_dist_flat)
        
        # Create scatter plot
        plt.scatter(original_dist_flat, mds_dist_flat, 
                  alpha=0.5, s=5)
        
        # Add trend line
        z = np.polyfit(original_dist_flat, mds_dist_flat, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(original_dist_flat), 
               p(np.sort(original_dist_flat)), 
               "r--", alpha=0.8, linewidth=2)
        
        plt.xlabel('Original Distances')
        plt.ylabel('MDS Distances')
        plt.title('Shepard Diagram')
        plt.grid(True, alpha=0.3)
        
        # Add correlation values
        plt.text(0.05, 0.95, 
               f'Pearson Correlation: {pearson_corr:.4f}\nSpearman Correlation: {spearman_corr:.4f}',
               transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plots.append({
            "title": "Shepard Diagram",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows how well the MDS preserves distances from the original space. Each point represents a pair of observations, plotting their original distance against their distance in the MDS space. Points along the diagonal indicate perfect distance preservation. The Pearson correlation of {pearson_corr:.4f} and Spearman correlation of {spearman_corr:.4f} quantify this relationship."
        })
    
    # Plot 4: Stress by Dimensions
    if X is not None or distances is not None:
        plt.figure(figsize=(10, 6))
        
        # Try a range of dimensions
        dimensions = range(1, min(10, len(embeddings)))
        stress_values = []
        
        for dim in dimensions:
            mds = MDS(n_components=dim, 
                    metric=metric, 
                    dissimilarity="precomputed" if distances is not None else "euclidean",
                    random_state=42)
            
            if distances is not None:
                mds.fit(distances)
            else:
                mds.fit(X)
                
            stress_values.append(mds.stress_)
        
        # Plot stress values
        plt.plot(dimensions, stress_values, 'o-', linewidth=2)
        plt.xlabel('Number of Dimensions')
        plt.ylabel('Stress')
        plt.title('Stress by Number of Dimensions')
        plt.grid(True, alpha=0.3)
        
        # Mark the current dimension
        plt.axvline(x=n_components, color='r', linestyle='--')
        plt.text(n_components+0.1, plt.ylim()[0] + 0.1*(plt.ylim()[1]-plt.ylim()[0]), 
               f'Current: {n_components}', color='r')
        
        # Find the "elbow" point
        diff = np.diff(stress_values)
        elbow_idx = np.argmax(diff[1:] - diff[:-1]) + 1 if len(diff) > 1 else 0
        elbow_dim = dimensions[elbow_idx + 1]
        
        plt.axvline(x=elbow_dim, color='g', linestyle=':')
        plt.text(elbow_dim+0.1, plt.ylim()[0] + 0.2*(plt.ylim()[1]-plt.ylim()[0]), 
               f'Elbow: {elbow_dim}', color='g')
        
        plt.tight_layout()
        
        plots.append({
            "title": "Stress by Dimensions",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows how the stress value decreases as the number of dimensions increases. Lower stress indicates better fit. The 'elbow' point at dimension {elbow_dim} suggests a good trade-off between dimensionality and fit quality. The current model uses {n_components} dimensions."
        })
    
    # Plot 5: Compare Metric and Non-metric MDS (if requested)
    if compare_methods and (X is not None or distances is not None):
        plt.figure(figsize=(15, 7))
        
        # Create two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Fit metric MDS
        metric_mds = MDS(n_components=2, 
                        metric=True, 
                        dissimilarity="precomputed" if distances is not None else "euclidean",
                        random_state=42)
        
        if distances is not None:
            metric_embedding = metric_mds.fit_transform(distances)
        else:
            metric_embedding = metric_mds.fit_transform(X)
            
        metric_stress = metric_mds.stress_
        
        # Fit non-metric MDS
        nonmetric_mds = MDS(n_components=2, 
                          metric=False, 
                          dissimilarity="precomputed" if distances is not None else "euclidean",
                          random_state=42)
        
        if distances is not None:
            nonmetric_embedding = nonmetric_mds.fit_transform(distances)
        else:
            nonmetric_embedding = nonmetric_mds.fit_transform(X)
            
        nonmetric_stress = nonmetric_mds.stress_
        
        # Plot metric MDS
        if color_values is not None:
            axes[0].scatter(metric_embedding[:, 0], metric_embedding[:, 1], 
                         c=color_values, cmap='viridis', 
                         s=50, alpha=0.8, edgecolors='w')
        else:
            axes[0].scatter(metric_embedding[:, 0], metric_embedding[:, 1], 
                         s=50, alpha=0.8, edgecolors='w')
            
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].set_title(f'Metric MDS (Stress: {metric_stress:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # Plot non-metric MDS
        if color_values is not None:
            axes[1].scatter(nonmetric_embedding[:, 0], nonmetric_embedding[:, 1], 
                         c=color_values, cmap='viridis', 
                         s=50, alpha=0.8, edgecolors='w')
        else:
            axes[1].scatter(nonmetric_embedding[:, 0], nonmetric_embedding[:, 1], 
                         s=50, alpha=0.8, edgecolors='w')
            
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        axes[1].set_title(f'Non-metric MDS (Stress: {nonmetric_stress:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Metric vs. Non-metric MDS",
            "img_data": get_base64_plot(),
            "interpretation": f"Compares metric MDS (left, stress={metric_stress:.4f}) and non-metric MDS (right, stress={nonmetric_stress:.4f}). Metric MDS preserves the actual distances, while non-metric MDS only preserves the ranking of distances. Non-metric MDS can better handle non-linear relationships but may distort actual distances. The method with lower stress generally provides a better representation of the data structure."
        })
    
    # Plot 6: Feature Contribution (if features are provided)
    if X is not None and feature_names is not None and len(feature_names) > 0:
        try:
            # Compute correlation between original features and MDS dimensions
            correlations = []
            for i in range(min(n_components, 3)):  # Limit to first 3 dimensions
                dim_corrs = []
                for j in range(X.shape[1]):
                    corr, _ = pearsonr(X[:, j], embeddings[:, i])
                    dim_corrs.append(corr)
                correlations.append(dim_corrs)
            
            # Create DataFrame for plotting
            corr_df = pd.DataFrame(np.array(correlations).T, 
                                 columns=[f'Dimension {i+1}' for i in range(len(correlations))],
                                 index=feature_names)
            
            # Plot heatmap
            plt.figure(figsize=(10, max(6, 0.3 * len(feature_names))))
            sns.heatmap(corr_df, cmap='coolwarm', center=0, annot=True, 
                       cbar_kws={'label': 'Correlation'})
            
            plt.title('Feature Contribution to MDS Dimensions')
            plt.tight_layout()
            
            plots.append({
                "title": "Feature Contribution Heatmap",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how each original feature correlates with the MDS dimensions. Strong positive or negative correlations (dark red or blue) indicate features that strongly influence a particular dimension. This helps interpret what the MDS dimensions represent in terms of the original features."
            })
        except Exception as e:
            print(f"Error generating feature contribution plot: {e}")
    
    # Plot 7: Cluster Visualization (if color represents clusters)
    if color_values is not None and len(np.unique(color_values)) <= 10:
        try:
            # Try to interpret color_values as clusters
            unique_clusters = np.unique(color_values)
            if len(unique_clusters) <= 10:  # Reasonable number of clusters
                plt.figure(figsize=(10, 8))
                
                # Create scatter plot with distinct colors for each cluster
                for cluster in unique_clusters:
                    cluster_points = embeddings[color_values == cluster]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              label=f'Cluster {cluster}', 
                              s=50, alpha=0.8, edgecolors='w')
                
                # Add cluster centroids
                for cluster in unique_clusters:
                    cluster_points = embeddings[color_values == cluster]
                    centroid = np.mean(cluster_points, axis=0)
                    plt.scatter(centroid[0], centroid[1], 
                              marker='X', s=200, 
                              edgecolors='k', linewidth=2,
                              label=f'Centroid {cluster}' if cluster == unique_clusters[0] else "",
                              color='black')
                
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.title('Cluster Visualization in MDS Space')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                plots.append({
                    "title": "Cluster Visualization",
                    "img_data": get_base64_plot(),
                    "interpretation": "Visualizes how different clusters are separated in the MDS space. Well-separated clusters indicate that the MDS has effectively captured the group structure in the data. Cluster overlap may indicate either similar groups or that more dimensions are needed to separate them properly."
                })
        except Exception as e:
            print(f"Error generating cluster visualization: {e}")
    
    # Plot 8: Pairwise MDS Dimensions (if more than 2 dimensions)
    if n_components > 2:
        try:
            # Limit to first 4 dimensions
            dims_to_plot = min(n_components, 4)
            
            # Create pairwise plots
            fig, axes = plt.subplots(dims_to_plot, dims_to_plot, 
                                   figsize=(12, 12))
            
            # Flatten axes for easier indexing
            axes_flat = axes.flatten()
            
            # Plot dimension pairs
            idx = 0
            for i in range(dims_to_plot):
                for j in range(dims_to_plot):
                    ax = axes[i, j]
                    
                    if i == j:  # Diagonal - plot histogram
                        ax.hist(embeddings[:, i], bins=20, alpha=0.7)
                        ax.set_title(f'Dimension {i+1}')
                    else:  # Off-diagonal - plot scatter
                        if color_values is not None:
                            ax.scatter(embeddings[:, j], embeddings[:, i], 
                                     c=color_values, cmap='viridis', 
                                     s=20, alpha=0.7)
                        else:
                            ax.scatter(embeddings[:, j], embeddings[:, i], 
                                     s=20, alpha=0.7)
                        
                        # Calculate correlation
                        corr, _ = pearsonr(embeddings[:, j], embeddings[:, i])
                        ax.text(0.05, 0.95, f'r = {corr:.2f}', 
                              transform=ax.transAxes, fontsize=8,
                              verticalalignment='top')
                    
                    # Only show axis labels on the edges
                    if i == dims_to_plot - 1:
                        ax.set_xlabel(f'Dimension {j+1}')
                    if j == 0:
                        ax.set_ylabel(f'Dimension {i+1}')
            
            plt.tight_layout()
            
            plots.append({
                "title": "Pairwise MDS Dimensions",
                "img_data": get_base64_plot(),
                "interpretation": "Shows pairwise relationships between MDS dimensions. Diagonal plots show the distribution of each dimension, while scatter plots show the relationship between dimension pairs. Low correlations between dimensions indicate that each one captures different aspects of the data structure."
            })
        except Exception as e:
            print(f"Error generating pairwise dimension plots: {e}")
            
    # Plot 9: Comparison with other dimensionality reduction techniques
    try:
        plt.figure(figsize=(15, 10))
        
        # Run PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(X)
        
        # Create subplots
        plt.subplot(1, 3, 1)
        if labels is not None:
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[plt.cm.tab10(i % 10) for i, _ in enumerate(mask)],
                    label=str(label),
                    alpha=0.7
                )
        else:
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
        plt.title('MDS')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        plt.subplot(1, 3, 2)
        if labels is not None:
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    c=[plt.cm.tab10(i % 10) for i, _ in enumerate(mask)],
                    label=str(label),
                    alpha=0.7
                )
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.title(f'PCA (Variance Explained: {pca.explained_variance_ratio_.sum():.2f})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.subplot(1, 3, 3)
        if labels is not None:
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(
                    tsne_result[mask, 0],
                    tsne_result[mask, 1],
                    c=[plt.cm.tab10(i % 10) for i, _ in enumerate(mask)],
                    label=str(label),
                    alpha=0.7
                )
        else:
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
        plt.title('t-SNE')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Add legend only to the first plot to avoid repetition
        if labels is not None:
            handles = [mpatches.Patch(color=plt.cm.tab10(i % 10), label=str(label)) 
                      for i, label in enumerate(np.unique(labels))]
            plt.figlegend(handles=handles, loc='lower center', ncol=min(5, len(np.unique(labels))))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the legend
        
        plots.append({
            "title": "Comparison of Dimensionality Reduction Techniques",
            "img_data": get_base64_plot(),
            "interpretation": "Compares MDS with PCA and t-SNE. PCA preserves global variance, t-SNE preserves local neighborhoods, and MDS preserves distances. Different clustering patterns across methods can reveal different aspects of your data structure."
        })
    except Exception as e:
        print(f"Error generating dimensionality reduction comparison plot: {e}")
    
    # Plot 10: Distance Preservation Analysis
    plt.figure(figsize=(12, 10))
    
    # Calculate the top K nearest neighbors in original and embedded spaces
    k = min(10, X.shape[0] - 1)
    
    # Get indices of k nearest neighbors in original space
    orig_neighbors = np.zeros((X.shape[0], k), dtype=int)
    for i in range(X.shape[0]):
        row_indices = np.argsort(distances[i])
        # Skip the first one (self)
        orig_neighbors[i] = row_indices[1:k+1]
    
    # Get indices of k nearest neighbors in embedded space
    emb_neighbors = np.zeros((X.shape[0], k), dtype=int)
    for i in range(X.shape[0]):
        row_indices = np.argsort(euclidean_distances(embeddings[i]).flatten())
        # Skip the first one (self)
        emb_neighbors[i] = row_indices[1:k+1]
    
    # Calculate neighbor preservation
    neighbor_overlap = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        overlap = np.intersect1d(orig_neighbors[i], emb_neighbors[i])
        neighbor_overlap[i] = len(overlap) / k
    
    # Plot neighbor preservation
    gs = gridspec.GridSpec(2, 2)
    
    # Histogram of neighbor preservation
    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(neighbor_overlap, bins=10, alpha=0.7)
    ax1.axvline(np.mean(neighbor_overlap), color='r', linestyle='--', 
               label=f'Mean: {np.mean(neighbor_overlap):.2f}')
    ax1.set_xlabel('Proportion of Preserved Neighbors')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Neighbor Preservation (k={k})')
    ax1.legend()
    
    # Scatter plot of neighbor preservation on the embedding
    ax2 = plt.subplot(gs[0, 1])
    scatter = ax2.scatter(
        embeddings[:, 0], 
        embeddings[:, 1], 
        c=neighbor_overlap, 
        cmap='viridis', 
        alpha=0.7
    )
    plt.colorbar(scatter, ax=ax2, label='Neighbor Preservation')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_title('Neighbor Preservation Mapped on Embedding')
    
    # Rank correlation of distances (Spearman)
    spearman_corr = stats.spearmanr(distances.flatten(), euclidean_distances(embeddings).flatten()).correlation
    
    # Preservation of rankings by distance
    ax3 = plt.subplot(gs[1, 0])
    plt.scatter(
        stats.rankdata(distances.flatten()),
        stats.rankdata(euclidean_distances(embeddings).flatten()),
        alpha=0.05
    )
    ax3.set_xlabel('Ranks of Original Distances')
    ax3.set_ylabel('Ranks of Embedded Distances')
    ax3.set_title(f'Rank Preservation (Spearman r = {spearman_corr:.4f})')
    
    # Calculate stress per point
    point_stress = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # Get all distances involving point i
        orig_dists = distances[i, :]
        emb_dists = euclidean_distances(embeddings[i]).flatten()
        # Calculate scaled squared error
        point_stress[i] = np.sum((orig_dists - emb_dists)**2) / np.sum(orig_dists**2)
    
    # Plot stress per point
    ax4 = plt.subplot(gs[1, 1])
    scatter = ax4.scatter(
        embeddings[:, 0], 
        embeddings[:, 1], 
        c=point_stress, 
        cmap='plasma', 
        alpha=0.7
    )
    plt.colorbar(scatter, ax=ax4, label='Point Stress')
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.set_title('Stress per Point')
    
    plt.tight_layout()
    
    plots.append({
        "title": "Distance Preservation Analysis",
        "img_data": get_base64_plot(),
        "interpretation": "Evaluates how well MDS preserves original data relationships. Top left: Distribution of neighbor preservation rates. Top right: Neighbor preservation mapped onto embedding (brighter = better). Bottom left: Preservation of distance rankings. Bottom right: Stress per point showing which points are poorly represented (brighter = worse)."
    })
    
    # Plot 11: Procrustes Analysis (if there are feature correlations)
    if feature_names is not None and X.shape[1] >= n_components:
        plt.figure(figsize=(12, 10))
        
        # Run PCA to get a linear projection for comparison
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)
        
        # Standardize both embeddings to allow comparison
        from scipy.spatial import procrustes
        
        # Procrustes transformation to align PCA with MDS
        mtx1, mtx2, disparity = procrustes(pca_result, embeddings)
        
        # Scatterplot of the aligned data
        plt.subplot(2, 2, 1)
        plt.scatter(mtx1[:, 0], mtx1[:, 1], alpha=0.7, label='PCA projection')
        plt.scatter(mtx2[:, 0], mtx2[:, 1], alpha=0.7, label='MDS (aligned)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'Procrustes Analysis (disparity = {disparity:.4f})')
        plt.legend()
        
        # Vector field showing the transformation
        plt.subplot(2, 2, 2)
        for i in range(X.shape[0]):
            plt.arrow(
                mtx1[i, 0], mtx1[i, 1],
                mtx2[i, 0] - mtx1[i, 0], mtx2[i, 0] - mtx1[i, 1],
                alpha=0.3, head_width=0.02, head_length=0.02
            )
        plt.scatter(mtx1[:, 0], mtx1[:, 1], alpha=0.7, label='PCA')
        plt.scatter(mtx2[:, 0], mtx2[:, 1], alpha=0.7, label='MDS')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Vector Field: PCA to MDS')
        plt.legend()
        
        # Top PCA components
        plt.subplot(2, 2, 3)
        top_n = min(10, len(feature_names))
        
        component_df = pd.DataFrame(
            pca.components_[:2, :top_n].T,
            index=feature_names[:top_n],
            columns=['PC1', 'PC2']
        )
        sns.heatmap(component_df, cmap='coolwarm', center=0, annot=True)
        plt.title('PCA Component Loadings')
        
        # Correlation between embedding dimensions and original features
        plt.subplot(2, 2, 4)
        
        # Calculate correlations
        corr_matrix = np.zeros((2, top_n))
        for i in range(2):
            for j in range(top_n):
                corr_matrix[i, j] = np.corrcoef(embeddings[:, i], X[:, j])[0, 1]
        
        corr_df = pd.DataFrame(
            corr_matrix,
            index=['MDS1', 'MDS2'],
            columns=feature_names[:top_n]
        )
        sns.heatmap(corr_df, cmap='coolwarm', center=0, annot=True)
        plt.title('MDS Dimension Correlations with Features')
        
        plt.tight_layout()
        
        plots.append({
            "title": "MDS vs. PCA Comparison",
            "img_data": get_base64_plot(),
            "interpretation": "Compares MDS with PCA projections. Top left: Aligned projections (lower disparity = more similar). Top right: Vector field showing how points move from PCA to MDS. Bottom left: PCA component loadings showing what features drive each dimension. Bottom right: Correlations between MDS dimensions and original features."
        })
    
    return plots 