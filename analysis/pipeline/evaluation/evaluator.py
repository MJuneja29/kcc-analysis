import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class ClusteringEvaluator:
    @staticmethod
    def evaluate(df, labels, features, counts_column='count'):
        """
        Evaluates clustering performance mathematically and commercially.
        :param df: The original dataframe containing queries and counts.
        :param labels: The cluster labels assigned (-1 is noise).
        :param features: The dense embeddings or UMAP coordinates used to calculate distances.
        :param counts_column: The column name indicating the frequency of the query.
        :return: Dict of metrics
        """
        # Ensure count array matches labels
        if counts_column in df.columns:
            counts = df[counts_column].values
        else:
            counts = np.ones(len(labels))
            
        total_volume = counts.sum()
        
        non_noise_mask = (labels != -1)
        valid_labels = labels[non_noise_mask]
        
        # Initialize default worst-case scores
        scores = {
            'silhouette': -1.0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0.0,
            'top_10_pct_coverage': 0.0,
            'median_cluster_size': 0.0,
            'noise_ratio_pct': 0.0,
            'total_clusters': 0
        }
        
        # 1. Noise Volume Ratio
        noise_volume = counts[labels == -1].sum()
        scores['noise_ratio_pct'] = round((noise_volume / total_volume) * 100, 2)
        
        # 2. Internal Quality Metrics (Only run if multiple valid clusters exist)
        num_clusters = len(set(valid_labels))
        scores['total_clusters'] = num_clusters
        
        if num_clusters > 1 and len(features[non_noise_mask]) > num_clusters:
            valid_features = features[non_noise_mask]
            
            # Silhouette: higher is better (-1 to 1) Measure of cluster tightness vs separation
            # Pass sample_size to speed up massive evaluations 
            sample_size = min(len(valid_features), 10000)
            scores['silhouette'] = round(silhouette_score(valid_features, valid_labels, metric='euclidean', sample_size=sample_size, random_state=42), 4)
            
            # Davies-Bouldin: lower is better (0 to inf) Average similarity between clusters and their most similar one
            scores['davies_bouldin'] = round(davies_bouldin_score(valid_features, valid_labels), 4)
            
            # Calinski-Harabasz: higher is better (Variance Ratio Criterion)
            scores['calinski_harabasz'] = round(calinski_harabasz_score(valid_features, valid_labels), 2)
            
        # 3. Commercial Efficiency Metrics (Coverage)
        if num_clusters > 0:
            # Calculate volume per cluster
            cluster_volumes = {}
            for label, count in zip(valid_labels, counts[non_noise_mask]):
                cluster_volumes[label] = cluster_volumes.get(label, 0) + count
                
            volumes_array = np.array(list(cluster_volumes.values()))
            
            # Median size
            scores['median_cluster_size'] = round(np.median(volumes_array), 2)
            
            # Top 10% Coverage
            top_10_pct_count = max(1, int(num_clusters * 0.10))
            top_clusters_volume = np.sort(volumes_array)[::-1][:top_10_pct_count].sum()
            
            # What % of ALL non-noise queries are answered by the top 10% of clusters?
            non_noise_volume = total_volume - noise_volume
            scores['top_10_pct_coverage'] = round((top_clusters_volume / max(1, non_noise_volume)) * 100, 2)
            
        return scores
