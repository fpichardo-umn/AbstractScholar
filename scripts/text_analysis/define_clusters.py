# -*- coding: utf-8 -*-
"""
STEP 03: Define Clusters

This script performs clustering of scientific articles based on their abstracts using k-means clustering algorithm and LSA. 

The process is as follows:
1. Load preprocessed data, including the document-term matrix and LSA components.
2. Calculate initial cluster coherence scores and identify high-scoring clusters.
3. Attempt to merge lower-scoring clusters if applicable.
4. Recalculate coherence scores for merged clusters.
5. Combine high-scoring original clusters with successful merged clusters.
6. Extract top terms for each final cluster.
7. Identify exemplar and most relevant articles for each cluster.
8. Save cluster information and sample data.

Clustering process:
The script uses a combination of k-means clustering and Latent Semantic Analysis (LSA) to group articles. 
It calculates coherence scores for clusters based on silhouette scores, explained sum of squares (ESS), 
and graph clustering coefficients. High-scoring clusters are preserved, while lower-scoring clusters 
may be merged if they meet certain similarity criteria.

Cluster merging:
Merging is attempted for clusters below the initial coherence threshold. The process uses cosine 
similarity to identify potential merge candidates. Merged clusters are only kept if they meet 
a new coherence threshold.

Final cluster selection:
The final set of clusters includes both high-scoring original clusters and any successful merged 
clusters that meet the coherence threshold.

Output:
The script generates several output files:
- 'cluster_info.txt': Contains information about each cluster, including top terms and exemplars.
- 'clustered_sample.txt': A subset of the original dataset with cluster assignments.
- 'cluster_abstracts.pickle': Pickled data containing detailed cluster information for further analysis.
Output:
- 'text_review_clusters.csv': A CSV file containing cluster IDs, top terms, and exemplars for user review.
  Users should add their assessment (I: Irrelevant, R: Relevant, B: Borderline) in the Assessment column.

NEXT STEP: Review the generated 'text_review_clusters.csv' file. Add your assessment (I, R, or B) 
for each cluster in the Assessment column. This will be used in subsequent analysis steps.

Requirements:
- Preprocessed document-term matrix and LSA components.
- Configuration file with necessary parameters.

Configuration entries:
   - 'preprocess_pickle': Path to the pickle file containing preprocessed data.
   - 'clusters_pickle': Path to the pickle file containing initial cluster data.
   - Other clustering parameters (e.g., similarity thresholds, coherence score calculations).

Created on Sun Apr 15 19:49:11 2018

@author: Felix Pichardo (original), [Your Name] (refactor)
"""

import sys
import os
import csv
import os.path as op
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import silhouette_samples
from collections import Counter
from kneed import KneeLocator
from math import log2, exp, sqrt

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_text import *

def make_centroids(cluster_names: List[int], cluster_idx: List[np.ndarray], doc_term_mat_xfm: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Calculate the centroids for each cluster.

    Args:
        cluster_names (List[int]): List of cluster names/IDs.
        cluster_idx (List[np.ndarray]): List of arrays containing indices for each cluster.
        doc_term_mat_xfm (np.ndarray): Document-term matrix.

    Returns:
        Dict[int, np.ndarray]: Dictionary of cluster centroids.
    """
    return {
        name: doc_term_mat_xfm[idx, :].mean(axis=0)
        for name, idx in zip(cluster_names, cluster_idx)
    }

def calc_cluster_silhouette_score(cluster_id: int, cluster_idx_map: Dict[int, np.ndarray], 
                                  doc_term_mat_dists: np.ndarray, num_articles: int) -> float:
    """
    Calculate the silhouette score for a given cluster.

    Args:
        cluster_id (int): The ID of the cluster.
        cluster_idx_map (Dict[int, np.ndarray]): Mapping of cluster IDs to article indices.
        doc_term_mat_dists (np.ndarray): Distance matrix between articles.
        num_articles (int): Total number of articles.

    Returns:
        float: The average silhouette score for the cluster.
    """
    cluster_articles_idx = cluster_idx_map[cluster_id]
    labels = np.zeros(num_articles)
    labels[cluster_articles_idx] = 1
    return silhouette_samples(doc_term_mat_dists, labels)[cluster_articles_idx].mean()

def calc_cluster_ess(cluster_id: int, cluster_idx_map: Dict[int, np.ndarray], 
                     centroids: Dict[int, np.ndarray], doc_term_mat_xfm: np.ndarray) -> float:
    """
    Calculate the log Explained Sum of Squares (ESS) for the given cluster.

    Args:
        cluster_id (int): The ID of the cluster.
        cluster_idx_map (Dict[int, np.ndarray]): Mapping of cluster IDs to article indices.
        centroids (Dict[int, np.ndarray]): Dictionary of cluster centroids.
        doc_term_mat_xfm (np.ndarray): Document-term matrix.

    Returns:
        float: The log Explained Sum of Squares (ESS) for the cluster.
    """
    cluster_articles_idx = cluster_idx_map[cluster_id]
    cluster_centroid = centroids[cluster_id]
    return np.log10(sum(
        np.sum((doc_term_mat_xfm[article] - cluster_centroid) ** 2)
        for article in cluster_articles_idx
    ) + 1)

def gaussian_dampen(x: float, mean: float, sigma: float) -> float:
    """
    Apply Gaussian dampening to a value.

    Args:
        x (float): The value to dampen.
        mean (float): The mean of the Gaussian distribution.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        float: The dampened value.
    """
    from math import exp
    return exp((-1/2) * ((x - mean)/sigma)**2)

def calc_cluster_coherence_score(cluster_id: int, cluster_sizes: Dict[int, int], 
                                 clusters_silhouette: Dict[int, float], clusters_ess: Dict[int, float], 
                                 num_articles: int, avg_cluster_score: Dict[int, float]) -> float:
    """
    Calculate the coherence score for a given cluster.

    Args:
        cluster_id (int): The ID of the cluster.
        cluster_sizes (Dict[int, int]): Dictionary of cluster sizes.
        clusters_silhouette (Dict[int, float]): Dictionary of cluster silhouette scores.
        clusters_ess (Dict[int, float]): Dictionary of cluster ESS scores.
        num_articles (int): Total number of articles.
        avg_cluster_score (Dict[int, float]): Dictionary of average cluster scores.

    Returns:
        float: The coherence score for the cluster.
    """
    
    num_clusters = len(cluster_sizes)
    cluster_size = cluster_sizes[cluster_id]
    mean_size = num_articles / num_clusters
    
    # Size component with penalty for very large clusters
    size_ratio = cluster_size / mean_size
    size_score = np.log1p(size_ratio) if size_ratio <= 1 else np.log1p(1 / size_ratio)
    
    # ESS component
    ess_score = np.log1p(sqrt(clusters_ess[cluster_id]))
    
    # Silhouette component
    silhouette_score = np.log1p(max(clusters_silhouette[cluster_id], 0))  # Ensure non-negative
    
    # Graph clustering coefficient component
    graph_coeff_score = np.log1p(avg_cluster_score[cluster_id])
    
    # Normalize scores
    scores = [size_score, ess_score, silhouette_score, graph_coeff_score]
    normalized_scores = [score / max(scores) for score in scores]
    
    # Calculate final score with equal weights
    coherence_score = sum(normalized_scores) / 4
    
    return coherence_score


def find_elbow_point(sorted_coherence_scores: List[Tuple[int, float]]) -> float:
    """
    Find the elbow point in the coherence scores.

    Args:
        sorted_coherence_scores (List[Tuple[int, float]]): List of tuples containing cluster IDs and their coherence scores.

    Returns:
        float: The elbow point (optimal coherence score threshold).
    """
    thresholds = np.linspace(
        start=max(sorted_coherence_scores, key=lambda x: x[1])[1],
        stop=min(sorted_coherence_scores, key=lambda x: x[1])[1],
        num=10
    )
    clusters_by_threshold = {
        thresh: [idx for idx, score in sorted_coherence_scores if score > thresh] 
        for thresh in thresholds
    }

    kneedle = KneeLocator(
        thresholds, 
        [len(clusters) for clusters in clusters_by_threshold.values()], 
        curve='convex', 
        direction='decreasing'
    )
    elbow_point = kneedle.elbow
    
    plot_elbow(thresholds, clusters_by_threshold, elbow_point)
    
    return elbow_point

def plot_elbow(thresholds: np.ndarray, clusters_by_threshold: Dict[float, List[int]], elbow_point: float):
    """
    Plot the elbow curve for coherence scores.

    Args:
        thresholds (np.ndarray): Array of threshold values.
        clusters_by_threshold (Dict[float, List[int]]): Dictionary mapping thresholds to clusters.
        elbow_point (float): The calculated elbow point.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, [len(clusters) for clusters in clusters_by_threshold.values()], 'b-', marker='o')
    plt.xlabel('Coherence Score Threshold')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters by Coherence Score Threshold')
    plt.axvline(x=elbow_point, color='r', linestyle='--')
    plt.show()

def merge_clusters(similarity_matrix: np.ndarray, threshold: float, centroids: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    merged_centroids = {}
    merged = set()
    centroid_keys = list(centroids.keys())
    
    # Create a list of all possible merge pairs with their similarity scores
    merge_candidates = []
    for i, cluster_i_key in enumerate(centroid_keys):
        for j, cluster_j_key in enumerate(centroid_keys[i+1:], start=i+1):
            if similarity_matrix[i, j] >= threshold:
                merge_candidates.append((cluster_i_key, cluster_j_key, similarity_matrix[i, j]))
    
    # Sort merge candidates by similarity score in descending order
    merge_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Perform merging
    for cluster_i_key, cluster_j_key, _ in merge_candidates:
        if cluster_i_key not in merged and cluster_j_key not in merged:
            new_centroid = (centroids[cluster_i_key] + centroids[cluster_j_key]) / 2
            new_cluster_id = f"merged_{cluster_i_key}_{cluster_j_key}"
            merged_centroids[new_cluster_id] = new_centroid
            merged.update([cluster_i_key, cluster_j_key])
    
    return merged_centroids

def update_cluster_info(clusters_info: Dict[int, Dict], merged_centroids: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    Update clusters_info with merged clusters.

    Args:
        clusters_info (Dict[int, Dict]): Original cluster information.
        merged_centroids (Dict[str, np.ndarray]): Merged cluster centroids.

    Returns:
        Dict[str, Dict]: Updated cluster information including merged clusters.
    """
    updated_info = {}
    for new_cluster_id, centroid in merged_centroids.items():
        cluster_ids_to_merge = map(int, new_cluster_id.split('_')[1:])
        info_to_merge = [clusters_info[cluster_id] for cluster_id in cluster_ids_to_merge]
        merged_info = {
            "centroids": centroid,
            "titles": np.hstack([info["titles"] for info in info_to_merge]),
            "indices": np.hstack([info["indices"] for info in info_to_merge])
        }
        updated_info[new_cluster_id] = merged_info
    return updated_info

def aggregate_avg_cluster_score(merged_clusters_info: Dict[str, Dict], avg_cluster_score: Dict[int, float]) -> Dict[str, float]:
    """
    Aggregate the average cluster score for merged clusters.

    Args:
        merged_clusters_info (Dict[str, Dict]): Information about merged clusters.
        avg_cluster_score (Dict[int, float]): Original average cluster scores.

    Returns:
        Dict[str, float]: Aggregated average cluster scores including merged clusters.
    """
    aggregated_scores = {}
    for merged_id, info in merged_clusters_info.items():
        if isinstance(merged_id, str) and '_' in merged_id:
            original_ids = [int(id) for id in merged_id.split('_')[1:]]
            aggregated_scores[merged_id] = np.mean([avg_cluster_score[orig_id] for orig_id in original_ids])
        else:
            aggregated_scores[merged_id] = avg_cluster_score[merged_id]
    return aggregated_scores

def get_cluster_top_terms(cluster_id, cluster_centroids, latent_sa, terms, num_terms=20):
    cluster_centroid = cluster_centroids[cluster_id]
    cluster_term_weights = np.abs(np.dot(cluster_centroid, latent_sa.components_))
    sorted_terms_idx = np.argsort(cluster_term_weights)[::-1][:num_terms]
    top_terms = [terms[idx] for idx in sorted_terms_idx]
    
    cnt_sub_in_terms = lambda x: sum([1 for term in top_terms if x in term])
    cleaned_top_terms = [term for term in top_terms if cnt_sub_in_terms(term) == 1]
    
    return cleaned_top_terms


def save_cluster_info(group_idx, centroids_infreq_top_terms, sample_titles, cluster_exemplars):
    df_cols = ['cluster', 'terms', 'titles', 'exemplar', 'order']
    df_data = [
        group_idx,
        [centroids_infreq_top_terms[idx] for idx in group_idx],
        [sample_titles[idx] for idx in group_idx],
        [cluster_exemplars[idx] for idx in group_idx],
        list(range(len(group_idx)))
    ]
    df_dict = dict(zip(df_cols, df_data))
    cluster_df = pd.DataFrame.from_dict(df_dict)
    cluster_csv = op.join('data', 'cluster_info.txt')
    cluster_df.to_csv(cluster_csv, sep='\t', encoding='iso-8859-1', index=False)


def save_clustered_sample(sample_titles, data):
    sample_cols = ['group', 'title', 'abstract', 'data_row']
    sample_group = []
    sample_title = []
    sample_abstract = []
    sample_article_idx = []
    for key, titles in sample_titles.items():
        sample_group.extend([str(key)] * len(titles))
        sample_title.extend(titles)
        data_idx = [data.loc[data.title == title].index[0] for title in titles]
        sample_abstract.extend(data.loc[data_idx].abstract)
        sample_article_idx.extend(data_idx)

    sample_data = [sample_group, sample_title, sample_abstract, sample_article_idx]
    sample_dict = dict(zip(sample_cols, sample_data))
    sample_df = pd.DataFrame.from_dict(sample_dict)
    sample_df.to_csv(op.join('data', 'clustered_sample.txt'), sep='\t', encoding='iso-8859-1', index=False)


def save_cluster_details(group_idx, centroids_infreq_top_terms, cluster_exemplars, cluster_most_relevant, sample_titles):
    with open(op.join('data', 'clustered_sample.txt'), 'w', encoding='iso-8859-1') as f:
        for order, cluster in enumerate(group_idx):
            f.write(f'Cluster {cluster}, Coherence Rank {order+1}:\n')
            f.write('Terms: ')
            f.write(', '.join(centroids_infreq_top_terms[cluster]))
            f.write('\n')

            f.write(f'Exemplar: *{cluster_exemplars[cluster]}*\n')
            f.write(f'Most relevant: *{cluster_most_relevant[cluster]}*\n')

            f.write('Articles:\n')
            for title in sample_titles[cluster]:
                if title not in [cluster_exemplars[cluster], cluster_most_relevant[cluster]]:
                    f.write(f'-{title[:145]}\n')
            f.write('\n')


def pickle_cluster_data(group_centroids, final_clusters_info, final_cluster_doc_term_mat_rows, 
                        final_cluster_names, final_cluster_titles, final_coherence_scores):
    cluster_pickle = op.join('data', 'cluster_abstracts.pickle')
    with open(cluster_pickle, 'wb') as p:
        pickle.dump([
            group_centroids, final_clusters_info, final_cluster_doc_term_mat_rows,
            final_cluster_names, final_cluster_titles, final_coherence_scores
        ], p)


def output_clusters_for_review(config: Dict, final_cluster_names: List, centroids_infreq_top_terms: Dict, cluster_exemplars: Dict, final_coherence_scores : Dict):
    """
    Output the list of clusters as a CSV file for user review.
    
    Args:
    config (Dict): User configuration dictionary
    final_cluster_names (List): List of final cluster names/IDs
    centroids_infreq_top_terms (Dict): Dictionary of infrequent top terms for each cluster
    cluster_exemplars (Dict): Dictionary of exemplar articles for each cluster
    final_coherence_scores (Dict): Dictionary of coherence scores for each cluster
    
    The function creates a CSV file with columns:
    Cluster ID, Top Terms, Exemplar, Assessment
    
    The Assessment column is left blank for the user to fill in with I, R, or B
    (Irrelevant, Relevant, or Borderline).
    """
    output_file = config.get('text_review_clusters_csv', 'data/text_review_clusters.csv')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cluster_id', 'Top Terms', 'Exemplar', 'Coherence', 'user_judgment'])
        
        for cluster_id in final_cluster_names:
            top_terms = ', '.join(centroids_infreq_top_terms.get(cluster_id, ['N/A']))
            exemplar = cluster_exemplars.get(cluster_id, 'N/A')
            coherence = np.round(final_coherence_scores.get(cluster_id, 'N/A'), 2)
            writer.writerow([cluster_id, top_terms, exemplar, coherence, ''])
    
    print(f"Cluster review file created: {output_file}")
    print("Please open this file and add your assessment (I, R, or B) in the user_judgment column for each cluster.")

def relative_gap_threshold(sorted_coherence_scores, gap_threshold=0.5):
    scores = [score for _, score in sorted_coherence_scores]
    gaps = [(scores[i] - scores[i+1]) / scores[i] for i in range(len(scores)-1)]
    if not gaps:
        return [sorted_coherence_scores[0][0]]
    cut_point = next((i for i, gap in enumerate(gaps) if gap > gap_threshold), len(sorted_coherence_scores)-1)
    return [idx for idx, _ in sorted_coherence_scores[:cut_point+1]]


# Load configuration and data
config = load_user_config()
data = load_data(config, clustered_data=True)
data.reset_index(drop=True, inplace=True)

# Get the configuration parameters
preprocess_pickle_filename = normalize_path(config.get('preprocess_pickle', './data/text_analysis/large_files/preprocessed_abstracts.pickle'))
clusters_pickle_filename = normalize_path(config.get('clusters_pickle', './data/text_analysis/large_files/clusters.pickle'))

# Load preprocessed abstract data
latent_sa, doc_term_mat_xfm, terms = load_from_pickle(preprocess_pickle_filename)
clusters, avg_cluster_score, G_strong = load_from_pickle(clusters_pickle_filename)

# Data validation
validate_data(data, required_columns=['title', 'abstract', 'cluster'])

# Prepare cluster data
data.cluster = data.cluster.astype(int)
get_cluster_idx = lambda x: data.query(f'cluster == {x}').index.values.astype(int)
get_cluster_titles = lambda x: data.query(f'cluster == {x}').title.values

cluster_names = [cluster for cluster in data.cluster.unique() if cluster != 999]
cluster_idx_list = [get_cluster_idx(name) for name in cluster_names]
cluster_idx_map = {name: get_cluster_idx(name) for name in cluster_names}
cluster_titles = {name: get_cluster_titles(name) for name in cluster_names}
cluster_sizes = {name: len(titles) for name, titles in cluster_titles.items()}

assert all(size > 0 for size in cluster_sizes.values()), "Found clusters with zero size"

# Calculate centroids and distances
centroids = make_centroids(cluster_names, cluster_idx_list, doc_term_mat_xfm)
doc_term_mat_dists = cosine_distances(doc_term_mat_xfm)

# Calculate silhouette scores and ESS
clusters_silhouette = {
    key: calc_cluster_silhouette_score(key, cluster_idx_map, doc_term_mat_dists, data.shape[0])
    for key in cluster_names
}

clusters_ess = {
    key: calc_cluster_ess(key, cluster_idx_map, centroids, doc_term_mat_xfm)
    for key in cluster_names
}

# Calculate coherence scores
clusters_coherence_score = {
    key: calc_cluster_coherence_score(
        key, cluster_sizes, clusters_silhouette, clusters_ess, 
        data.shape[0], avg_cluster_score
    )
    for key in cluster_names
}

sorted_coherence_scores = sorted(clusters_coherence_score.items(), key=lambda x: x[1], reverse=True)

# Prepare clusters info
clusters_info = {
    cluster_id: {
        "centroids": centroids[cluster_id],
        "titles": cluster_titles[cluster_id],
        "indices": get_cluster_idx(cluster_id),
    }
    for cluster_id, _ in sorted_coherence_scores
}

# Find elbow point and threshold clusters
if (len(sorted_coherence_scores) > 5) or (data.shape[0] > 100):
    score_thresh = find_elbow_point(sorted_coherence_scores)
    thresholded_idx = [idx for idx, score in sorted_coherence_scores if score > score_thresh]
else:
    thresholded_idx = relative_gap_threshold(sorted_coherence_scores)

filtered_sorted_coherence_scores = {idx: score for idx, score in sorted_coherence_scores if idx not in thresholded_idx}

# Start of potential merging process
print("Checking for potential cluster merging...")

if len(filtered_sorted_coherence_scores) > 1:
    # Calculate similarity threshold and matrix
    centroid_dict = {cluster_id: centroids[cluster_id] for cluster_id in filtered_sorted_coherence_scores.keys()}
    centroid_list = list(centroid_dict.values())
    similarity_matrix = cosine_similarity(centroid_list)
    similarity_matrix_flat = np.round(similarity_matrix[np.tril_indices_from(similarity_matrix)], 7)
    similarity_threshold = np.percentile(similarity_matrix_flat[similarity_matrix_flat < 1], 90)

    # Merge clusters
    merged_centroids = merge_clusters(similarity_matrix, similarity_threshold, centroid_dict)
else:
    merged_centroids = {}

if merged_centroids:
    print(f"Merging {len(merged_centroids)} clusters...")
    merged_clusters_info = update_cluster_info(clusters_info, merged_centroids)

    # Update cluster information after merging
    updated_cluster_idx_map = {cluster_id: info['indices'] for cluster_id, info in merged_clusters_info.items()}
    updated_cluster_doc_term_mat_rows = {
        cluster_id: doc_term_mat_xfm[info['indices'], :] 
        for cluster_id, info in merged_clusters_info.items()
    }

    updated_cluster_sizes_dict = {cluster_id: len(info['titles']) * 2 for cluster_id, info in merged_clusters_info.items()}

    # Recalculate scores for updated clusters
    updated_clusters_silhouette = {
        cluster_id: calc_cluster_silhouette_score(
            cluster_id, updated_cluster_idx_map, doc_term_mat_dists, data.shape[0]
        )
        for cluster_id in merged_clusters_info.keys()
    }

    combined_centroids = {**centroids, **merged_centroids}
    updated_clusters_ess = {
        cluster_id: calc_cluster_ess(
            cluster_id, updated_cluster_idx_map, combined_centroids, doc_term_mat_xfm
        )
        for cluster_id in updated_cluster_idx_map.keys()
    }

    updated_avg_cluster_score = aggregate_avg_cluster_score(merged_clusters_info, avg_cluster_score)

    updated_clusters_coherence_score = {
        cluster_id: calc_cluster_coherence_score(
            cluster_id, updated_cluster_sizes_dict, updated_clusters_silhouette,
            updated_clusters_ess, data.shape[0], updated_avg_cluster_score
        )
        for cluster_id in updated_clusters_silhouette.keys()
    }

    update_sorted_coherence_scores = sorted(
        updated_clusters_coherence_score.items(), key=lambda x: x[1], reverse=True
    )

    # Check if any merged clusters meet the threshold
    if len(update_sorted_coherence_scores) > 1:
        new_score_thresh = find_elbow_point(update_sorted_coherence_scores)
        updated_thresholded_idx = [idx for idx, score in update_sorted_coherence_scores if score > new_score_thresh]
    else:
        updated_thresholded_idx = [update_sorted_coherence_scores[0][0]]

    if updated_thresholded_idx:
        print("Some merged clusters meet coherence threshold. Combining with original high-scoring clusters.")
    
        # Identify which original clusters were not merged
        unmerged_clusters = set(thresholded_idx) - set(filtered_sorted_coherence_scores.keys())
        
        # Combine unmerged original clusters with thresholded merged clusters
        final_cluster_ids = list(unmerged_clusters) + updated_thresholded_idx
        
        final_centroids = {}
        final_clusters_info = {}
        final_cluster_doc_term_mat_rows = {}
        final_cluster_titles = {}
        final_coherence_scores = []
        
        for cluster_id in final_cluster_ids:
            if cluster_id in unmerged_clusters:
                final_centroids[cluster_id] = centroids[cluster_id]
                final_clusters_info[cluster_id] = clusters_info[cluster_id]
                final_cluster_doc_term_mat_rows[cluster_id] = doc_term_mat_xfm[clusters_info[cluster_id]["indices"], :]
                final_cluster_titles[cluster_id] = cluster_titles[cluster_id]
                final_coherence_scores.append((cluster_id, clusters_coherence_score[cluster_id]))
            else:
                final_centroids[cluster_id] = combined_centroids[cluster_id]
                final_clusters_info[cluster_id] = merged_clusters_info[cluster_id]
                final_cluster_doc_term_mat_rows[cluster_id] = updated_cluster_doc_term_mat_rows[cluster_id]
                final_cluster_titles[cluster_id] = merged_clusters_info[cluster_id]["titles"]
                final_coherence_scores.append((cluster_id, updated_clusters_coherence_score[cluster_id]))
        
        final_coherence_scores = sorted(final_coherence_scores, key=lambda x: x[1], reverse=True)
        final_cluster_names = [item[0] for item in sorted(final_coherence_scores, key=lambda x: x[1], reverse=True)]
    else:
        print("No merged clusters meet coherence threshold. Keeping only original high-scoring clusters.")
        final_centroids = {k: centroids[k] for k in thresholded_idx}
        final_clusters_info = {k: clusters_info[k] for k in thresholded_idx}
        final_cluster_doc_term_mat_rows = {k: doc_term_mat_xfm[clusters_info[k]["indices"], :] for k in thresholded_idx}
        final_cluster_titles = {k: cluster_titles[k] for k in thresholded_idx}
        final_coherence_scores = [(k, clusters_coherence_score[k]) for k in thresholded_idx]
        final_cluster_names = [item[0] for item in sorted(final_coherence_scores, key=lambda x: x[1], reverse=True)]
else:
    print("No clusters were merged. Keeping only original high-scoring clusters.")
    final_centroids = {k: centroids[k] for k in thresholded_idx}
    final_clusters_info = {k: clusters_info[k] for k in thresholded_idx}
    final_cluster_doc_term_mat_rows = {k: doc_term_mat_xfm[clusters_info[k]["indices"], :] for k in thresholded_idx}
    final_cluster_titles = {k: cluster_titles[k] for k in thresholded_idx}
    final_coherence_scores = [(k, clusters_coherence_score[k]) for k in thresholded_idx]
    final_cluster_names = [item[0] for item in sorted(final_coherence_scores, key=lambda x: x[1], reverse=True)]

print("Merging process completed.")

# Calculate infrequent top terms
print("Calculating infrequent top terms for clusters...")
centroids_top_terms = {}
all_infreq_top_centroid_terms = Counter()
for cluster_id in final_centroids.keys():
    top_terms = get_cluster_top_terms(cluster_id, final_centroids, latent_sa, terms)[:50]
    centroids_top_terms[cluster_id] = top_terms
    all_infreq_top_centroid_terms.update(top_terms)

max_cnt = sorted(all_infreq_top_centroid_terms.items(), key=lambda x: x[1], reverse=True)[0][1]
max_freq = max_cnt // 2 if max_cnt > 7 else max_cnt

centroids_infreq_top_terms = {}
for cluster_id in final_centroids.keys():
    top_terms = get_cluster_top_terms(cluster_id, final_centroids, latent_sa, terms)[:50]
    centroids_infreq_top_terms[cluster_id] = [term for term in top_terms if all_infreq_top_centroid_terms[term] < max_freq][:20]

# Check for infrequent terms
print("Checking for infrequent terms in clusters...")

if sum([results != [] for _, results in centroids_infreq_top_terms.items()]) == 0:
    # if all infrequent terms are empty, use the original top terms
    centroids_infreq_top_terms = centroids_top_terms
else:
    # if some infrequent terms are present, use them
    print("Some infrequent terms found. Using them for clusters.")
    for cluster_id in final_centroids.keys():
        if centroids_infreq_top_terms[cluster_id] == []:
            centroids_infreq_top_terms[cluster_id] = centroids_top_terms[cluster_id]

# Prepare final data
print("Preparing final data...")
group_centroids = {key: final_centroids[key] for key, _ in final_coherence_scores}

cmp_arts_centroid = lambda x: np.dot(final_cluster_doc_term_mat_rows[x], group_centroids[x])
art_dot_centroid = {idx: cmp_arts_centroid(idx) for idx in group_centroids.keys()}

get_top_art_idx = lambda x: np.unique(np.concatenate([
    np.where((art_dot_centroid[x] >= 0.9))[0],
    np.where(art_dot_centroid[x] > np.round(np.median(art_dot_centroid[x]), 5))[0]
]))

idx_top_arts = {idx: get_top_art_idx(idx).tolist() for idx in group_centroids.keys()}
sorted_dot_centroids = {key: np.sort(dot.copy()) for key, dot in art_dot_centroid.items()}

# Resample data
group_idx = list(idx_top_arts.keys())
while True:
    group_lens = [len(idx_top_arts[idx]) for idx in group_idx]
    if 1 not in group_lens:
        break
    idx_to_remove = [idx for idx in group_idx if len(idx_top_arts[idx]) == 1]
    for idx in idx_to_remove:
        group_idx.remove(idx)

resample_len = 2 * min(group_lens)
to_resample = [idx for idx, g_len in enumerate(group_lens) if g_len > resample_len]

# Resample data
group_idx = list(idx_top_arts.keys())
while True:
    group_lens = [len(idx_top_arts[idx]) for idx in group_idx]
    if 1 not in group_lens:
        break
    idx_to_remove = [idx for idx in group_idx if len(idx_top_arts[idx]) == 1]
    for idx in idx_to_remove:
        group_idx.remove(idx)

resample_len = 2 * min(group_lens)
to_resample = [idx for idx in group_idx if len(idx_top_arts[idx]) > resample_len]

resampled_idx = idx_top_arts.copy()
for group in to_resample:
    vals_to_find = sorted_dot_centroids[group][-resample_len:]
    new_idx = []
    for val in vals_to_find:
        new_idx.extend(np.where(art_dot_centroid[group] == val)[0].tolist())
    resampled_idx[group] = list(set(new_idx))

sample_titles = {
    cluster: [final_cluster_titles[cluster][i] for i in resampled_idx[cluster]] 
    for cluster in resampled_idx.keys()
}
sample_dtm = {
    cluster: final_cluster_doc_term_mat_rows[cluster][resampled_idx[cluster], :] 
    for cluster in resampled_idx.keys()
}

article_top_concepts = get_article_top_concepts(doc_term_mat_xfm)

# Get exemplars and most relevant articles
get_exemplar_idx = lambda x: np.where(art_dot_centroid[x] == art_dot_centroid[x].max())
cluster_exemplars = {
    cluster: final_cluster_titles[cluster][get_exemplar_idx(cluster)[0][0]] 
    for cluster in group_idx
}

avg_centroid = doc_term_mat_xfm.mean(axis=0)
centroids_to_avg = {
    cluster_id: np.dot(final_centroids[cluster_id], avg_centroid) 
    for cluster_id in group_centroids.keys()
}

cmp_cluster_to_avg = lambda centroid: np.dot(
    final_cluster_doc_term_mat_rows[centroid], centroids_to_avg[centroid]
)
cluster_cmp_avg = {centroid: cmp_cluster_to_avg(centroid) for centroid in centroids_to_avg.keys()}

get_relevant_idx = lambda x: np.where(cluster_cmp_avg[x] == cluster_cmp_avg[x].max())
cluster_most_relevant = {
    cluster: final_cluster_titles[cluster][get_relevant_idx(cluster)[0][0]]
    for cluster in cluster_cmp_avg.keys()
}

# Save results
print("Saving results...")
save_cluster_info(group_idx, centroids_infreq_top_terms, sample_titles, cluster_exemplars)
save_clustered_sample(sample_titles, data)
save_cluster_details(group_idx, centroids_infreq_top_terms, cluster_exemplars, cluster_most_relevant, sample_titles)

# Pickle other variables
print("Pickling final data...")
pickle_cluster_data(group_centroids, final_clusters_info, final_cluster_doc_term_mat_rows, 
                    final_cluster_names, final_cluster_titles, final_coherence_scores)

print("Clustering process completed.")

# After all cluster processing is complete
output_clusters_for_review(config, final_cluster_names, centroids_infreq_top_terms, cluster_exemplars, dict(final_coherence_scores))
