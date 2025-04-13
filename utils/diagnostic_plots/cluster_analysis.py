"""Cluster Analysis diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from itertools import combinations
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

def generate_cluster_analysis_plots(X, feature_names=None, max_clusters=10):
    """Generate diagnostic plots for Cluster Analysis
    
    Args:
        X: Features (numpy array)
        feature_names: Names of the features (list of strings)
        max_clusters: Maximum number of clusters to evaluate
        
    Returns:
        List of dictionaries with plot information
    """
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
    
    # Ensure we have the right number of feature names
    if len(feature_names) != X.shape[1]:
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    plots = []
    
    # Plot 1: Elbow Method for determining the optimal number of clusters
    plt.figure(figsize=(10, 6))
    distortions = []
    for i in range(1, min(11, X.shape[0])):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)
    
    plt.plot(range(1, min(11, X.shape[0])), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    
    # Try to find the "elbow" point
    if len(distortions) > 2:
        # Calculate second derivative to find where the curve bends
        deltas = np.diff(distortions, 2)
        if len(deltas) > 0:
            elbow_point = np.argmin(deltas) + 2  # +2 because of double diff and 0-indexing
            plt.vlines(elbow_point, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashed', label=f'Elbow point = {elbow_point}')
            plt.legend()
    
    plots.append({
        "title": "Elbow Method Plot",
        "img_data": get_base64_plot(),
        "interpretation": "The elbow method helps determine the optimal number of clusters by plotting the distortion " +
                        "(sum of squared distances from each point to its assigned center) against the number of clusters. " +
                        "The 'elbow' in the curve indicates the point where adding more clusters doesn't significantly reduce distortion. " +
                        "The red dashed line (if present) indicates the estimated optimal number of clusters."
    })
    
    # Plot 2: Silhouette Analysis
    # Choose a reasonable number of clusters based on elbow method or maximum 5 for demonstration
    if len(distortions) > 2 and np.argmin(np.diff(distortions, 2)) + 2 < 6:
        n_clusters = np.argmin(np.diff(distortions, 2)) + 2
    else:
        n_clusters = min(5, X.shape[0] - 1)
    
    plt.figure(figsize=(12, 8))
    
    # Initialize silhouette scores array
    silhouette_avg_list = []
    
    # Range of n_clusters (up to the lesser of max_clusters or data size minus one)
    n_range = range(2, min(max_clusters + 1, X.shape[0]))
    
    # Calculate silhouette scores for each number of clusters
    for n in n_range:
        clusterer = KMeans(n_clusters=n, random_state=42)
        cluster_labels = clusterer.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.plot(list(n_range), silhouette_avg_list, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.grid(True)
    
    # Mark the maximum silhouette score
    if silhouette_avg_list:
        best_n = list(n_range)[np.argmax(silhouette_avg_list)]
        plt.vlines(best_n, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashed', 
                   label=f'Best n_clusters = {best_n}')
        plt.legend()
    
    plots.append({
        "title": "Silhouette Analysis",
        "img_data": get_base64_plot(),
        "interpretation": "Silhouette analysis measures how well samples are clustered by calculating the distance between " +
                         "clusters. The score ranges from -1 to 1, with higher values indicating better defined clusters. " +
                         "This plot shows the average silhouette score for different numbers of clusters. " +
                         "The peak (marked with a red dashed line) suggests the optimal number of clusters."
    })
    
    # Plot 3: Detailed Silhouette Plot (for the optimal number of clusters)
    if silhouette_avg_list:
        best_n = list(n_range)[np.argmax(silhouette_avg_list)]
        kmeans = KMeans(n_clusters=best_n, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        plt.figure(figsize=(12, 8))
        
        # Compute the silhouette scores for each sample
        silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
        
        y_lower = 10
        for i in range(best_n):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
            ith_cluster_silhouette_vals.sort()
            
            size_cluster_i = ith_cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / best_n)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_vals,
                              facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        # The silhouette_score is the average of the sample silhouette coefficient
        plt.axvline(x=silhouette_avg_list[best_n-2], color="red", linestyle="--")
        plt.title(f"Detailed Silhouette Plot for {best_n} Clusters")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster Label")
        
        # Set y-axis limits to show all clusters
        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.grid(True)
        
        plots.append({
            "title": "Detailed Silhouette Plot",
            "img_data": get_base64_plot(),
            "interpretation": "This plot shows the silhouette coefficient for each sample in the optimal cluster configuration. " +
                             "Each cluster is represented by a different color, and the width of each silhouette indicates how well " +
                             "that sample fits in its cluster. Wide silhouettes indicate well-matched samples, while narrow ones " +
                             "may be in the wrong cluster. The red dashed line shows the average silhouette score. " +
                             "Good clustering shows consistently wide silhouettes with similar widths across clusters."
        })
    
    # Plot 4: K-means Clustering Result (2D visualization)
    # If more than 2 dimensions, use the two most informative dimensions
    if X.shape[1] > 2:
        # Use silhouette scores to find the most informative features
        # This is a simple approach - PCA would be more robust but more complex
        feature_scores = []
        for i, j in combinations(range(X.shape[1]), 2):
            X_subset = X_scaled[:, [i, j]]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_subset)
            try:
                score = silhouette_score(X_subset, cluster_labels)
                feature_scores.append((i, j, score))
            except:
                continue
        
        if feature_scores:
            best_pair = max(feature_scores, key=lambda x: x[2])
            best_features = [best_pair[0], best_pair[1]]
            feature_names_2d = [feature_names[best_features[0]], feature_names[best_features[1]]]
            X_2d = X_scaled[:, best_features]
        else:
            # If silhouette scoring fails, just take the first two dimensions
            X_2d = X_scaled[:, :2]
            feature_names_2d = feature_names[:2]
    else:
        X_2d = X_scaled
        feature_names_2d = feature_names
    
    # Perform K-means clustering on the 2D data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X_2d)
    centers = kmeans.cluster_centers_
    
    plt.figure(figsize=(10, 8))
    
    # Plot the scatter points with cluster colors
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.8)
    
    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    
    # Add feature names as axis labels
    plt.xlabel(feature_names_2d[0])
    plt.ylabel(feature_names_2d[1])
    plt.title(f'K-means Clustering ({n_clusters} Clusters)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    
    plots.append({
        "title": "K-means Clustering Result",
        "img_data": get_base64_plot(),
        "interpretation": "This plot shows the result of K-means clustering projected onto the two most informative dimensions. " +
                         "Each point represents a sample, colored by its assigned cluster. The red X marks indicate " +
                         "cluster centers. Well-defined clusters appear as distinct, separated groups with minimal overlap. " +
                         f"The chosen dimensions are {feature_names_2d[0]} and {feature_names_2d[1]}, which were selected " +
                         "to best display the cluster separation."
    })
    
    # Plot 5: Hierarchical Clustering Dendrogram
    plt.figure(figsize=(12, 8))
    
    # Generate the linkage matrix
    Z = linkage(X_scaled, method='ward')
    
    # Plot the dendrogram
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    
    # Draw a horizontal line at the level where the best number of clusters is formed
    if silhouette_avg_list:
        best_n = list(n_range)[np.argmax(silhouette_avg_list)]
        dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='lastp', p=best_n)
        plt.axhline(y=Z[-best_n+1, 2], color='r', linestyle='--', 
                   label=f'Cut for {best_n} clusters')
        plt.legend()
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    plots.append({
        "title": "Hierarchical Clustering Dendrogram",
        "img_data": get_base64_plot(),
        "interpretation": "A dendrogram visualizes hierarchical clustering, showing how samples are grouped together " +
                         "at different levels of similarity. The height of each branch represents the distance between " +
                         "the merged clusters. At the bottom are individual samples, which progressively merge as you move up. " +
                         "The red dashed line (if present) shows where to cut the tree to obtain the optimal number of clusters. " +
                         "Clusters are formed by all branches below this line."
    })
    
    # Plot 6: Feature Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Create a heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                xticklabels=feature_names, yticklabels=feature_names)
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Correlation Heatmap",
        "img_data": get_base64_plot(),
        "interpretation": "This heatmap shows the correlation between each pair of features. " +
                         "Values range from -1 (perfect negative correlation, dark blue) to 1 (perfect positive correlation, dark red), " +
                         "with 0 indicating no correlation (white). Strong correlations suggest redundant features that may " +
                         "unnecessarily influence the clustering. When features are highly correlated, you might consider " +
                         "removing some to simplify the model while maintaining its effectiveness."
    })
    
    return plots 