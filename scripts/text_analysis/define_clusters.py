# -*- coding: utf-8 -*-
"""
STEP 03: Define Clusters

Is there a way I can determine the top features/terms for each cluster in while the data was decomposed?

Assuming latent_sa = TruncatedSVD(n_components=k) for some k, the obvious way to get
term weights makes use of the fact that LSA/SVD is a linear transformation, i.e.,
each row of latent_sa.components_ is a weighted sum of the input terms, and you can
multiply that with the cluster centroids from k-means.


DESCRIPTION
This script performs clustering of scientific articles based on their abstracts using k-means clustering algorithm. The output is a set of clusters, where each cluster is represented by a set of terms that best describe the cluster, an exemplar article that is most representative of the cluster, and a list of titles of articles that belong to the cluster.

To use this script, you will need to have a dataset of scientific articles with their titles and abstracts in a CSV file format. You will also need to have the following Python packages installed: pandas, numpy, sklearn, and nltk.

To run the script, simply specify the path to the CSV file containing the article data, the number of clusters to create, and the number of terms to extract per cluster. The script will then perform k-means clustering on the abstracts, extract the top terms for each cluster, and output the results to three files: "cluster_info.txt", "clustered_sample.txt", and "cluster_samples.txt".

"cluster_info.txt" contains the list of clusters and their associated terms, exemplars, and titles. "clustered_sample.txt" contains a subset of the original dataset, with each article assigned to its corresponding cluster. "cluster_samples.txt" contains a detailed description of each cluster, including its coherence rank, top terms, exemplar, and most relevant article, as well as a list of article titles belonging to the cluster.

This script provides a simple and effective way to cluster scientific articles based on their abstracts, allowing for easier organization and analysis of large datasets.

Created on Sun Apr 15 19:49:11 2018

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
# This imports utility and basic modules functions from the 'scripts' directory
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_text import *

# Script specific imports
from math import exp, log2, sqrt
from pprint import pprint
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import silhouette_samples
from collections import Counter
from kneed import KneeLocator


def get_cluster_data(data):
    """
    Retrieves cluster data from the given input data.

    Args:
        data (DataFrame): The input data containing cluster information.

    Returns:
        tuple: A tuple containing the following elements:
            - cluster_names (list): A list of unique cluster names.
            - cluster_idx_list (list): A list of cluster indices.
            - cluster_idx_map (dict): A dictionary mapping cluster names to indices.
            - cluster_titles (dict): A dictionary mapping cluster names to titles.
    """
    cluster_names = list(data.cluster.unique())
    cluster_names.remove(999)
    cluster_idx_list = [get_cluster_idx(name) for name in cluster_names]
    cluster_idx_map = {name: get_cluster_idx(name) for name in cluster_names}
    cluster_titles = {name: get_cluster_titles(name) for name in cluster_names}
    
    return cluster_names, cluster_idx_list, cluster_idx_map, cluster_titles


def get_cluster_top_terms(cluster_id, cluster_centroids, latent_sa, terms, num_terms=20):
    """
    Get the top terms for a given cluster.

    Args:
        cluster_id (int): The ID of the cluster.
        cluster_centroids (numpy.ndarray): The centroids of the clusters.
        latent_sa (numpy.ndarray): The latent semantic analysis components.
        terms (list): The list of terms.
        num_terms (int, optional): The number of top terms to return. Defaults to 20.

    Returns:
        list: The cleaned top terms for the cluster.
    """
    # Get centroid and term weights for cluster
    cluster_centroid = cluster_centroids[cluster_id]
    cluster_term_weights = np.abs(np.dot(cluster_centroid, latent_sa.components_)) # need absolute values for the weights because of the sign indeterminacy in LSA
    
    # Get the indices of the top terms
    sorted_terms_idx = np.argsort(cluster_term_weights)[::-1][:num_terms]
    top_terms = [terms[idx] for idx in sorted_terms_idx]
    
    # Clean the top terms
    is_unique_substring = lambda x: sum([1 for term in top_terms if x in term]) == 1
    cleaned_top_terms = [term for term in top_terms if is_unique_substring(term)]

    return cleaned_top_terms


def make_centroids(cluster_names, cluster_idx, doc_term_mat_xfm):
    """
    Calculate the centroids for each cluster.

    Args:
        cluster_names (list): A list of cluster names.
        cluster_idx (list): A list of cluster indices.
        doc_term_mat_xfm (numpy.ndarray): A 2D array representing the document-term matrix.

    Returns:
        dict: A dictionary containing the cluster centroids, where the keys are the cluster names and the values are the centroids.
    """
    # Calculate the centroids for each cluster by taking the mean along axis=0
    cluster_centroids = {
        name: doc_term_mat_xfm[idx, :].mean(axis=0)
        for name, idx in zip(cluster_names, cluster_idx)
    }

    return cluster_centroids


def calc_cluster_silhouette_score(cluster_id, cluster_idx_map, doc_term_mat_dists, num_articles):
    """
    Calculate the silhouette score for a given cluster.

    Args:
        cluster_id (int): The ID of the cluster.
        cluster_idx_map (dict): A dictionary mapping cluster IDs to article indices.
        doc_term_mat_dists (numpy.ndarray): The distance matrix between articles.
        num_articles (int): The total number of articles.

    Returns:
        float: The average silhouette score for the cluster.
    """
    
    cluster_articles_idx = cluster_idx_map[cluster_id]
    labels = np.zeros(num_articles)
    labels[cluster_articles_idx] = 1
    
    return silhouette_samples(doc_term_mat_dists, labels)[cluster_articles_idx].mean()


def mean(l):
    """Calculate the mean of a list of numbers."""
    return sum(l)/len(l)


def calc_cluster_ess(cluster_id, cluster_idx_map, centroids, doc_term_mat_xfm):
    """Calculates the Explained Sum of Squares (ESS) for the given cluster.

    Args:
        cluster_id (int): The ID of the cluster.
        cluster_idx_map (dict): A dictionary mapping cluster IDs to article indices.
        centroids (numpy.ndarray): An array of cluster centroids.
        doc_term_mat_xfm (numpy.ndarray): A matrix representing the document-term matrix.

    Returns:
        float: The Explained Sum of Squares (ESS) for the cluster.

    Notes:
        The ESS is used as a measure of how well the cluster centroid represents the articles in the cluster.
        It is calculated as the sum of squared differences between each article's document-term matrix (DTM)
        and the cluster centroid.
    """
    cluster_articles_idx = cluster_idx_map[cluster_id]
    cluster_centroid = centroids[cluster_id]
    
    # Calculate ESS by summing squared differences
    ess = sum(
        np.sum((doc_term_mat_xfm[article] - cluster_centroid) ** 2)
        for article in cluster_articles_idx
    )

    return ess


def gaussian_dampen(x, mean, sigma):
    """Dampen the score for values that are not close to the mean."""
    return exp((-1/2) * ((x - mean)/sigma)**2)


def calc_cluster_coherence_score(cluster_id, cluster_sizes, clusters_silhouette, clusters_ess, num_articles, avg_cluster_score):
    """Return coherence score for a given cluster.

    This function evaluates the coherence of a given cluster using the silhouette score,
    explained sum of squares, and the size of the cluster.

    Parameters:
    - cluster_id: The ID of the cluster for which to calculate the coherence score.
    - cluster_sizes: A list containing the number of articles in each cluster.
    - clusters_silhouette: A dictionary containing the silhouette scores for each cluster.
    - clusters_ess: A dictionary containing the explained sum of squares for each cluster.
    - num_articles (int): The total number of articles.
    - avg_cluster_score: A dictionary containing the average cluster score for each cluster.

    Returns:
    - coherence_score: The coherence score for the given cluster.

    How it works:
    - Calculates the mean number of articles per cluster.
    - The function uses a dampening function to adjust the score for clusters that are not close to the mean size.
    - The dampened cluster size is multiplied by the cluster size to give more weight to larger clusters.
    - The function calculates the coherence score by multiplying the dampened cluster size, the square root of the explained sum of squares, the silhouette score raised to the power of 3, and the average cluster score.
    - The coherence score is rounded to 2 decimal places and returned.
    """

    num_clusters = len(cluster_sizes)
    cluster_size = cluster_sizes[cluster_id]

    mean = num_articles/num_clusters

    # Larger clusters should have a better chance than smaller clusters but not too large
    # Apply gaussian dampening to the cluster size
    sigma = int(log2(exp((cluster_size + 2) / 2)))
    cluster_size_dampen = gaussian_dampen(cluster_sizes[cluster_id], mean, sigma)
    cluster_dampened_size = cluster_size_dampen * cluster_size
    
    # Retrieve necessary scores for the coherence calculation
    graph_clustering_coeff = avg_cluster_score[cluster_id]
    cluster_silhouette = clusters_silhouette[cluster_id]
    cluster_sqrt_ess = sqrt(clusters_ess[cluster_id])

    coherence_score = (cluster_dampened_size + cluster_sqrt_ess) * cluster_silhouette**3 * graph_clustering_coeff

    return round(coherence_score, 2)


def find_elbow_point(sorted_coherence_scores):
    """
    Finds the elbow point in a plot of the number of clusters against the coherence score threshold.

    Args:
        sorted_coherence_scores (list): A list of tuples containing the coherence scores sorted in descending order.

    Returns:
        int: The elbow point, which is the optimal coherence score threshold.
    """
    # Gradual threshold adjustment
    thresholds = np.linspace(start=max(sorted_coherence_scores, key=lambda x: x[1])[1], stop=min(sorted_coherence_scores, key=lambda x: x[1])[1], num=10)
    clusters_by_threshold = {thresh: [idx for idx, score in sorted_coherence_scores if score > thresh] for thresh in thresholds}

    # Assuming thresholds and num_clusters are x and y axes respectively
    kneedle = KneeLocator(thresholds, [len(clusters) for clusters in clusters_by_threshold.values()], curve='convex', direction='decreasing')
    elbow_point = kneedle.elbow
    print(f"Elbow Point at Threshold: {elbow_point}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, [len(clusters) for clusters in clusters_by_threshold.values()], 'b-', marker='o')
    plt.xlabel('Coherence Score Threshold')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters by Coherence Score Threshold')
    plt.axvline(x=elbow_point, color='r', linestyle='--')
    plt.show()

    return elbow_point


def calculate_similarity_threshold(filtered_sorted_coherence_scores, centroids):
    """
    Calculates the similarity threshold (90th percentile) and similarity matrix based on the given filtered sorted coherence scores and centroids.

    Args:
        filtered_sorted_coherence_scores (dict): A dictionary containing the filtered and sorted coherence scores for each cluster.
        centroids (dict): A dictionary containing the centroids for each cluster.

    Returns:
        tuple: A tuple containing the similarity threshold, similarity matrix, and centroid_dict.
    """
    centroid_dict = {cluster_id: centroids[cluster_id] for cluster_id in filtered_sorted_coherence_scores.keys()}
    centroid_list = [centroids[cluster_id] for cluster_id in filtered_sorted_coherence_scores.keys()]
    similarity_matrix = cosine_similarity(centroid_list)
    similarity_threshold = np.percentile(np.sort(similarity_matrix.flatten()), 90)
    return similarity_threshold, similarity_matrix, centroid_dict 


def merge_cluster_centroids(centroid_a, centroid_b):
    """Merge two clusters by averaging their centroids.

    Args:
        centroid_a (numpy.ndarray): Centroid of the first cluster.
        centroid_b (numpy.ndarray): Centroid of the second cluster.

    Returns:
        numpy.ndarray: The centroid of the merged cluster.
    """
    return (centroid_a + centroid_b) / 2


def find_clusters_to_merge(similarity_matrix, threshold, centroids):
    """Find and merge clusters based on the similarity threshold.

    Args:
        similarity_matrix (numpy.ndarray): Matrix of pairwise similarities between clusters.
        threshold (float): Similarity threshold for merging clusters.
        centroids (dict): Dictionary of cluster IDs to their centroids.

    Returns:
        dict: Dictionary of merged cluster IDs to their new centroids.
    """
    
    merged_centroids = {}
    merged = set()

    centroid_keys = list(centroids.keys())

    for i, cluster_i_key in enumerate(centroid_keys):
        if i in merged:
            continue
        for j, cluster_j_key in enumerate(centroid_keys[i+1:], start=i+1):
            if j in merged:
                continue
            # Check if the similarity is above the threshold
            if similarity_matrix[i, j] >= threshold:
                new_centroid = merge_cluster_centroids(centroids[cluster_i_key], centroids[cluster_j_key])
                new_cluster_id = f"merged_{cluster_i_key}_{cluster_j_key}"
                merged_centroids[new_cluster_id] = new_centroid
                merged.update([i, j])

    return merged_centroids


def merge_clusters_info(info_a, info_b):
    """Merge information from two clusters."""
    merged_info = {
        "centroids": (info_a["centroids"] + info_b["centroids"]) / 2,
        "titles": np.hstack((info_a["titles"], info_b["titles"])),
        "indices": np.hstack((info_a["indices"], info_b["indices"]))
    }
    return merged_info


def update_cluster_info(clusters_info, merged_centroids):
    """Update clusters_info with merged clusters."""
    updated_info = {}
    for new_cluster_id, centroid in merged_centroids.items():
        cluster_ids_to_merge = map(int, new_cluster_id.split('_')[1:])  # Extract original cluster IDs from the new_cluster_id
        info_to_merge = [clusters_info[cluster_id] for cluster_id in cluster_ids_to_merge]
        merged_info = merge_clusters_info(*info_to_merge)
        merged_info["centroids"] = centroid  # Use the centroid from merged_centroids
        updated_info[new_cluster_id] = merged_info
        
    return updated_info


def aggregate_avg_cluster_score(merged_clusters_info, avg_cluster_score):
    """Aggregate the average cluster score for merged clusters."""
    aggregated_scores = {}
    for merged_id, info in merged_clusters_info.items():
        if type(merged_id) == str and '_' in merged_id:
            original_ids = [int(id) for id in merged_id.split('_')[1:]]  # Extract original cluster IDs
            aggregated_scores[merged_id] = np.mean([avg_cluster_score[orig_id] for orig_id in original_ids])
        else:
            aggregated_scores[merged_id] = avg_cluster_score[merged_id]
            
    return aggregated_scores



###
#   START
###

# Load preprocessed data
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
validate_data(data, required_columns = ['title', 'abstract', 'cluster'])









##Get Cluster Data
data.cluster        = data.cluster.astype(int)
get_cluster_idx     = lambda x: data.query('cluster == {}'.format(x)).index.values.astype(int)
get_cluster_titles  = lambda x: data.query('cluster == {}'.format(x)).title.values

#Cluster names without solo
cluster_names   = list(data.cluster.unique())
cluster_names.remove(999)
cluster_names   = [cluster for cluster in cluster_names]

#Cluster title idx and titles with maps
cluster_idx_list    = [get_cluster_idx(name) for name in cluster_names]
cluster_idx_map     = {name:get_cluster_idx(name) for name in cluster_names}
cluster_titles      = [get_cluster_titles(name) for name in cluster_names]
cluster_titles      = {name:titles for name, titles in zip(cluster_names, cluster_titles)}
cluster_sizes        = [len(cluster_titles[cluster]) for cluster in cluster_names]
assert 0 not in cluster_sizes

cluster_doc_term_mat_rows = {name:doc_term_mat_xfm[idx, :]
                        for name, idx in zip(cluster_names, cluster_idx_list)}

centroids = make_centroids(cluster_names, cluster_idx_list, doc_term_mat_xfm)

##Calc silhouette and select best clusters
doc_term_mat_dists = cosine_distances(doc_term_mat_xfm)

clusters_silhouette  = {}
for key in cluster_doc_term_mat_rows.keys():
    clusters_silhouette[key] = calc_cluster_silhouette_score(key, cluster_idx_map, doc_term_mat_dists, data.shape[0])

clusters_silhouette = {key: calc_cluster_silhouette_score(key, cluster_idx_map, doc_term_mat_dists, data.shape[0]) for key in cluster_doc_term_mat_rows.keys()}

clusters_ess = {}
for key in clusters_silhouette.keys():
    clusters_ess[key] = calc_cluster_ess(key, cluster_idx_map, centroids, doc_term_mat_xfm)

#Calc coherence score
clusters_coherence_score  = {}
for key in clusters_silhouette.keys():
    clusters_coherence_score[key] =  calc_cluster_coherence_score(key, cluster_sizes, clusters_silhouette, clusters_ess, data.shape[0], avg_cluster_score)

sorted_coherence_scores = sorted(clusters_coherence_score.items(), key=lambda x: x[1], reverse = True)


score_thresh = find_elbow_point(sorted_coherence_scores)

thresholded_idx = [idx for idx, score in sorted_coherence_scores if score > score_thresh]
filtered_sorted_coherence_scores = {idx: score for idx, score in sorted_coherence_scores if idx not in thresholded_idx}

similarity_threshold, similarity_matrix, centroid_dict = calculate_similarity_threshold(filtered_sorted_coherence_scores, centroids)

clusters_info = {
    cluster_id: {
        "centroids": centroids[cluster_id],  # Centroids for each cluster
        "titles": cluster_titles[cluster_id],  # Titles within each cluster
        "indices": get_cluster_idx(cluster_id),  # Document indices within each cluster
        # Add other relevant information as needed
    }
    for cluster_id in filtered_sorted_coherence_scores.keys()
}

clusters_info_orig = {
    cluster_id: {
        "centroids": centroids[cluster_id],  # Centroids for each cluster
        "titles": cluster_titles[cluster_id],  # Titles within each cluster
        "indices": get_cluster_idx(cluster_id),  # Document indices within each cluster
        # Add other relevant information as needed
    }
    for cluster_id in [item[0] for item in sorted_coherence_scores]
}





merged_centroids = find_clusters_to_merge(similarity_matrix, similarity_threshold, centroid_dict)

# Update the clusters_info with the merged clusters
merged_clusters_info = update_cluster_info(clusters_info, merged_centroids)

# Assuming 'merged_clusters_info' contains the updated indices and titles for merged clusters
updated_cluster_idx_map = {cluster_id: info['indices'] for cluster_id, info in merged_clusters_info.items()}
updated_cluster_doc_term_mat_rows = {cluster_id: doc_term_mat_xfm[info['indices'], :] for cluster_id, info in merged_clusters_info.items()}


updated_cluster_sizes_dict = {cluster_id: len(info['titles']) for cluster_id, info in merged_clusters_info.items()}

updated_clusters_silhouette = {
    cluster_id: calc_cluster_silhouette_score(cluster_id, updated_cluster_idx_map, doc_term_mat_dists, data.shape[0])
    for cluster_id in merged_clusters_info.keys()
}

# Assuming 'merged_centroids' contains centroids for merged clusters and 'centroids' for unmerged
combined_centroids = {**centroids, **merged_centroids}
updated_clusters_ess = {
    cluster_id: calc_cluster_ess(cluster_id, updated_cluster_idx_map, combined_centroids, doc_term_mat_xfm)
    for cluster_id in updated_cluster_idx_map.keys()
}

updated_avg_cluster_score = aggregate_avg_cluster_score(merged_clusters_info, avg_cluster_score)


updated_clusters_coherence_score = {
    cluster_id: calc_cluster_coherence_score(
        cluster_id,
        updated_cluster_sizes_dict,  # Pass the dictionary instead of the list
        updated_clusters_silhouette,
        updated_clusters_ess,
        data.shape[0],
        updated_avg_cluster_score
    )
    for cluster_id in updated_clusters_silhouette.keys()
}

update_sorted_coherence_scores = sorted(updated_clusters_coherence_score.items(), key=lambda x: x[1], reverse = True)

# Gradual threshold adjustment
thresholds = np.linspace(start=max(update_sorted_coherence_scores, key=lambda x: x[1])[1], stop=min(update_sorted_coherence_scores, key=lambda x: x[1])[1], num=10)
clusters_by_threshold = {thresh: [idx for idx, score in update_sorted_coherence_scores if score > thresh] for thresh in thresholds}

# Assuming thresholds and num_clusters are your x and y axes respectively
kneedle = KneeLocator(thresholds, [len(clusters) for clusters in clusters_by_threshold.values()], curve='convex', direction='decreasing')

# The elbow point
elbow_point = kneedle.elbow

print(f"Elbow Point at Threshold: {elbow_point}")

# Plotting for visualization
plt.figure(figsize=(8, 6))
plt.plot(thresholds, [len(clusters) for clusters in clusters_by_threshold.values()], 'b-', marker='o')
plt.xlabel('Coherence Score Threshold')
plt.ylabel('Number of Clusters')
plt.title('Number of Clusters by Coherence Score Threshold')
plt.axvline(x=elbow_point, color='r', linestyle='--')
plt.show()

score_thresh = elbow_point

updated_thresholded_idx = [idx for idx, score in update_sorted_coherence_scores if score > score_thresh]


# Include unmerged clusters by adding their info from the original clusters_info
for cluster_id in thresholded_idx:
    merged_centroids[cluster_id] = centroids[cluster_id]
    merged_clusters_info[cluster_id] = clusters_info_orig[cluster_id]
    
updated_cluster_doc_term_mat_rows = {cluster_id: doc_term_mat_xfm[info['indices'], :] for cluster_id, info in merged_clusters_info.items()}

updated_cluster_names = list(merged_clusters_info.keys())
updated_cluster_titles = {name: merged_clusters_info[name]["titles"] for name in updated_cluster_names}


# Sort by coherence
all_coherences = [(item[0], item[1]) for item in sorted_coherence_scores if item[0] in thresholded_idx] + [(item[0], item[1]) for item in update_sorted_coherence_scores if item[0] in updated_thresholded_idx]
sorted_all_coherences = sorted(all_coherences, key=lambda x: x[1], reverse = True)


# Use merged centroids for group definitions
group_centroids = {key:merged_centroids[key] for key in [item[0] for item in sorted_all_coherences]}

# Updated lambda function to calculate dot product using new centroids
cmp_arts_centroid = lambda x: np.dot(updated_cluster_doc_term_mat_rows[x], group_centroids[x])

# Updated dictionary to hold the dot product results
art_dot_centroid = {idx: cmp_arts_centroid(idx) for idx in group_centroids.keys()}

# Updated method to get top article indices based on the updated centroids
get_top_art_idx = lambda x: np.unique(np.concatenate([np.where((art_dot_centroid[x] >= .9))[0], np.where(art_dot_centroid[x] > art_dot_centroid[x].mean())[0]]))

idx_top_arts = {idx: get_top_art_idx(idx).tolist() for idx in group_centroids.keys()}

sorted_dot_centroids = {key:np.sort(dot.copy()) for key, dot in art_dot_centroid.items()}


#Resample data
group_idx =  list(idx_top_arts.keys())
redo_groups = True
while redo_groups:
    group_lens = [len(idx_top_arts[idx]) for idx in group_idx]
    #Remove group that has only one article
    if 1 in group_lens:
        idx_to_remove = [idx for idx in group_idx if len(idx_top_arts[idx]) == 1]
        for idx in idx_to_remove:
            group_idx.remove(idx)
        continue

    break

resample_len = 2 * min(group_lens)
to_resample = []
for idx, g_len in enumerate(group_lens):
    if g_len > resample_len:
        to_resample.append(group_idx[idx])

resampled_idx = idx_top_arts.copy()
for group in to_resample:
    vals_to_find = sorted_dot_centroids[group][-resample_len:]
    new_idx = []
    for val in vals_to_find:
        new_idx.extend(np.where(art_dot_centroid[group] == val)[0].tolist())
    resampled_idx[group] = new_idx
    resampled_idx[group] = list(set(resampled_idx[group]))


#Make training data
sample_titles = {cluster: [updated_cluster_titles[cluster][i] for i in resampled_idx[cluster]] for cluster in resampled_idx.keys()}
sample_dtm = {cluster: updated_cluster_doc_term_mat_rows[cluster][resampled_idx[cluster], :] for cluster in resampled_idx.keys()}

article_top_concepts = get_article_top_concepts(doc_term_mat_xfm)


#Get exemplar
get_exemplar_idx = lambda x: np.where(art_dot_centroid[x] == art_dot_centroid[x].max())
cluster_exemplars = {cluster: updated_cluster_titles[cluster][get_exemplar_idx(cluster)[0][0]] for cluster in group_idx}


##Get most relevant from group
avg_centroid = doc_term_mat_xfm.mean(axis=0)
centroids_to_avg = {cluster_id: np.dot(merged_centroids[cluster_id], avg_centroid) for cluster_id in group_centroids.keys()}

cmp_cluster_to_avg = lambda centroid: np.dot(updated_cluster_doc_term_mat_rows[centroid], centroids_to_avg[centroid])
cluster_cmp_avg = {centroid:cmp_cluster_to_avg(centroid) for centroid in centroids_to_avg.keys()}

get_relevant_idx = lambda x: np.where(cluster_cmp_avg[x] == cluster_cmp_avg[x].max())
cluster_most_relevant = {cluster:updated_cluster_titles[cluster][get_relevant_idx(cluster)[0][0]]
                        for cluster in cluster_cmp_avg.keys()}


#Get cluster top, infreq terms
centroids_top_terms = {}
all_infreq_top_centroid_terms = Counter()
for cluster_id in group_centroids.keys():
    top_terms = get_cluster_top_terms(cluster_id, merged_centroids, latent_sa, terms)[:50]
    centroids_top_terms[cluster_id] = top_terms
    all_infreq_top_centroid_terms.update(top_terms)

max_cnt = sorted(all_infreq_top_centroid_terms.items(), key= lambda x: x[1], reverse = True)[0][1]
max_freq = max_cnt//2 if max_cnt > 7 else max_cnt

centroids_infreq_top_terms = {}
for cluster_id in group_centroids.keys():
    top_terms = get_cluster_top_terms(cluster_id, group_centroids, latent_sa, terms)[:50]
    centroids_infreq_top_terms[cluster_id] = [term for term in top_terms if all_infreq_top_centroid_terms[term] < max_freq][:20]


df_cols = ['cluster', 'terms', 'titles', 'exemplar', 'order']
cluster_terms = [centroids_infreq_top_terms[idx] for idx in group_idx]
cluster_titles_g = [sample_titles[idx] for idx in group_idx]
cluster_exemplars_for_df = [cluster_exemplars[idx] for idx in group_idx]
cluster_order = [idx for idx, _ in enumerate(group_idx)]
df_data = [group_idx, cluster_terms, cluster_titles_g, cluster_exemplars_for_df, cluster_order]
df_dict = {key:val for key, val in zip(df_cols, df_data)}
cluster_df = pd.DataFrame.from_dict(df_dict)

cluster_csv = op.join('data', 'cluster_info.txt')
cluster_df.to_csv(cluster_csv, sep = '\t', encoding='iso-8859-1', index = False)

sample_cols = ['group', 'title', 'abstract', 'data_row']
sample_group = []
sample_title = []
sample_abstract = []
sample_article_idx = []
for key in sample_titles.keys():
    titles = sample_titles[key]
    groups = '{} '.format(key) * len(titles)
    sample_group.extend(groups.split(' ')[:-1])
    sample_title.extend(titles)
    data_idx = [data.loc[data.title == title].index[0] for title in titles]
    sample_abstract.extend(data.loc[data_idx].abstract)
    sample_article_idx.extend(data_idx)

sample_data = [sample_group, sample_title, sample_abstract, sample_article_idx]
sample_dict = {key:val for key, val in zip(sample_cols, sample_data)}
sample_df = pd.DataFrame.from_dict(sample_dict)

sample_df.to_csv(op.join('data', 'clustered_sample.txt'), sep = '\t', encoding='iso-8859-1', index = False)

with open(op.join('data', 'clustered_sample.txt'), 'w', encoding='iso-8859-1') as f:
    for order, cluster in enumerate(group_idx):
        f.write('Cluster {}, Coherence Rank {}:\n'.format(cluster, order+1))
        f.write('Terms: ')
        cluster_terms_to_save = centroids_infreq_top_terms[cluster]
        f.write(', '.join(cluster_terms_to_save))
        f.write('\n')

        exemplar = cluster_exemplars[cluster]
        f.write('Exemplar: *')
        f.write(exemplar)
        f.write('*')
        f.write('\n')

        relv = cluster_most_relevant[cluster]
        f.write('Most relevant: *')
        f.write(relv)
        f.write('*')
        f.write('\n')

        f.write('Articles:\n')
        cluster_titles_to_save = sample_titles[cluster]
        for title in cluster_titles_to_save:
            if title not in [exemplar, relv]:
                f.write('-')
                f.write(title[:145])
                f.write('\n')
        f.write('\n')


# Pickle other variables
cluster_pickle = op.join('data', 'cluster_abstracts.pickle')
with open(cluster_pickle, 'wb') as p:
    pickle.dump([group_centroids, merged_clusters_info, updated_cluster_doc_term_mat_rows, updated_cluster_names, updated_cluster_titles, sorted_all_coherences], p)

