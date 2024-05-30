# -*- coding: utf-8 -*-
"""
STEP 03: Define Clusters

Is there a way I can determine the top features/terms for each cluster in while the data was decomposed?

Assuming lsa = TruncatedSVD(n_components=k) for some k, the obvious way to get
term weights makes use of the fact that LSA/SVD is a linear transformation, i.e.,
each row of lsa.components_ is a weighted sum of the input terms, and you can
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

import pickle
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, log2, sqrt
from pprint import pprint
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from kneed import KneeLocator

def get_article_top_concepts(dtm_lsa):

    art_top_concepts = {art:[] for art in range(dtm_lsa.shape[0])}
    top_num = 5
    for i in range(len(dtm_lsa)):
       top_topics = np.argsort(dtm_lsa[i,:])[::-1][:top_num]
       top_topics_str = ' '.join(str(t) for t in top_topics)
       art_top_concepts[i] = [int(top) for top in top_topics_str.split() if int(top) != 0]

    return art_top_concepts


def get_cluster_top_terms(cluster_id, cluster_centroids, lsa, terms, num_terms = 20):

    #Get centroid
    cluster_centroid = cluster_centroids[cluster_id]

    #Get term weights for cluster
    cluster_term_weights = np.dot(cluster_centroid, lsa.components_)
    cluster_term_weights = np.abs(cluster_term_weights) #need absolute values for the weights because of the sign indeterminacy in LSA
    sorted_terms_idx = np.argsort(cluster_term_weights)[::-1][:num_terms]
    top_terms = [terms[idx] for idx in sorted_terms_idx]

    cnt_sub_in_terms = lambda x: sum([1 for term in top_terms if x in term])
    cleaned_top_terms = [term for term in top_terms if cnt_sub_in_terms(term) == 1]

    return cleaned_top_terms


def get_terms_for_concept(terms, components, concept_idx, num_terms = 20):

    comp = components[concept_idx]
    terms_in_comp = zip(terms, comp)
    sorted_terms = sorted(terms_in_comp, key = lambda x: x[1], reverse = True) [:num_terms]
    sorted_terms = [term[0] for term in sorted_terms]

    cnt_sub_in_terms = lambda x: sum([1 for term in sorted_terms if x in term])
    cleaned_sorted_terms = [term for term in sorted_terms if cnt_sub_in_terms(term) == 1]

    return cleaned_sorted_terms


def make_centroids(cluster_names, cluster_idx, dtm_lsa):

    #Get cluster centroids
    #Take mean along axis=0 of the dtm rows for the cluster articles
    cluster_centroids = {name:dtm_lsa[idx, :].mean(axis=0)
                        for name, idx in zip(cluster_names, cluster_idx)}

    return cluster_centroids


def calc_silhouette_for_cluster(cluster_id, cluster_idx_map, dist_dtm, num_articles):
    """Returns the silhouette for a given cluters

    Uses distance matrix to calculate the mean distances between cluster_id
    (from list of cluster_idx) and all the articles in the cluster and between
    all the articles outside the cluster. Returns si
    """

    cluster_articles_idx = cluster_idx_map[cluster_id]
    non_idx_mask = np.ones(num_articles, dtype=bool)
    non_idx_mask[cluster_articles_idx] = False
    not_cluster_articles_idx = data.index[non_idx_mask]

    dist_cluster = []
    dist_other = []
    for article in cluster_articles_idx:
        idx_to_del = np.where(cluster_articles_idx == article)
        cluster_idx_not_article = np.delete(cluster_articles_idx.copy(), idx_to_del)

        dist_cluster.append(dist_dtm[article][cluster_idx_not_article].mean())
        dist_other.append(dist_dtm[article][not_cluster_articles_idx].mean())

    ai = mean(dist_cluster)
    bi = min(dist_other)
#    sum(dist_other)/len(dist_other)

    si = (bi-ai)/max(bi, ai)

    return si


def mean(l):

    return sum(l)/len(l)


def calc_cluster_ess(cluster_id, cluster_idx_map, centroids, dtm_lsa):
    """Returns the Explained Sum of Squares (ESS) for the cluster

    Use ESS because the centroid is defined as the mean of the article's
    dtm rows. The higher, the better
    """

    cluster_articles_idx = cluster_idx_map[cluster_id]
    cluster_centroid = centroids[cluster_id]

    squares = []
    for article in cluster_articles_idx:
        article_dtm = dtm_lsa[article, :]
        centroid_diff = article_dtm - cluster_centroid
        squares.append(centroid_diff.dot(centroid_diff))

    return sum(squares)


def gaussian_dampen(x, mean, sigma):
    """Dampens xs that are not close enough to mean
    """

    return exp((-1/2) * ((x - mean)/sigma)**2)


def calc_cluster_coherence_score(cluster_id, clusters_len, clusters_silhouette, clusters_ess, data, dtm_lsa, avg_cluster_score):
    """Return coherence score for a given cluster.

    Evalutes the coherence of a given cluster using the silhouette score,
    explained sum of squares, and the size of the cluster.

    A good cluster should more than a few items in it but not too many.
    The gaussian_dampen functions dampens the score for values that are not
    close to the num_articles/num_clusters because a good set of clusters would
    divide the space all equally well. Bigger clusters are boosted more than smaller
    ones via the sigma function
    """

    num_articles = data.shape[0]
    num_clusters = len(clusters_len)
    cluster_size = clusters_len[cluster_id]

    mean = num_articles/num_clusters

    #I want the larger clusters to have a better chance than smaller clusters
    #But not too large
    sigma = int(log2(exp((cluster_size+2)/2)))
    cluster_size_dampen    = gaussian_dampen(clusters_len[cluster_id], mean, sigma)
    cluster_dampened_size = cluster_size_dampen * cluster_size

    graph_clustering_coeff = avg_cluster_score[cluster_id]

    cluster_silhouette  = clusters_silhouette[cluster_id]
    cluster_sqrt_ess    = sqrt(clusters_ess[cluster_id])

    coherence_score = (cluster_dampened_size + cluster_sqrt_ess) * cluster_silhouette**3 * graph_clustering_coeff

    return round(coherence_score, 2)


def merge_clusters_centroids(centroid_a, centroid_b):
    """Merge two clusters by averaging their centroids.

    Args:
        centroid_a (numpy.ndarray): Centroid of the first cluster.
        centroid_b (numpy.ndarray): Centroid of the second cluster.

    Returns:
        numpy.ndarray: The centroid of the merged cluster.
    """
    merged_centroid = (centroid_a + centroid_b) / 2
    return merged_centroid


def find_clusters_to_merge(similarity_matrix, threshold, centroids):
    """Find and merge clusters based on the similarity threshold.

    Args:
        similarity_matrix (numpy.ndarray): Matrix of pairwise similarities between clusters.
        threshold (float): Similarity threshold for merging clusters.
        centroids (dict): Dictionary of cluster IDs to their centroids.

    Returns:
        dict: Dictionary of merged cluster IDs to their new centroids.
    """
    num_clusters = len(centroids)
    merged_centroids = {}
    merged = set()

    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            # Check if the similarity is above the threshold and the clusters haven't been merged already
            if similarity_matrix[i, j] >= threshold and i not in merged and j not in merged:
                cluster_i_key = list(centroids.keys())[i]
                cluster_j_key = list(centroids.keys())[j]
                new_centroid = merge_clusters_centroids(centroids[cluster_i_key], centroids[cluster_j_key])
                new_cluster_id = f"merged_{cluster_i_key}_{cluster_j_key}"
                merged_centroids[new_cluster_id] = new_centroid
                merged.update([i, j])

    return merged_centroids


def merge_clusters_info(info_a, info_b):
    """Merge information from two clusters."""
    
    merged_info = {
        "centroids": (info_a["centroids"] + info_b["centroids"]) / 2,
        "titles": np.hstack((info_a["titles"], info_b["titles"])),
        "indices": np.hstack((info_a["indices"], info_b["indices"])),
        # Merge other attributes as needed
    }
    return merged_info


def update_clusters_info(clusters_info, merged_centroids):
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

preprocess_pickle = op.join('data', 'preprocessed_abstracts.pickle')
with open(preprocess_pickle, 'rb') as p:
    lsa, dtm_lsa, terms = pickle.load(p)

clusters_pickle = op.join('data', 'clusters.pickle')
with open(clusters_pickle, 'rb') as p:
    clusters, avg_cluster_score, G_strong = pickle.load(p)

data_for_edist = op.join('data', 'clustered_data.txt')
data = pd.read_csv(data_for_edist, sep = '\t', encoding='iso-8859-1', index_col = None)

##Get Cluster Data
data.cluster        = data.cluster.astype(int)
get_cluster_idx     = lambda x: data.query('cluster == {}'.format(x)).index.values
get_cluster_titles  = lambda x: data.query('cluster == {}'.format(x)).Title.values

#Cluster names without solo
cluster_names   = list(data.cluster.unique())
cluster_names.remove(999)
cluster_names   = [cluster for cluster in cluster_names]

#Cluster title idx and titles with maps
cluster_idx_list    = [get_cluster_idx(name) for name in cluster_names]
cluster_idx_map     = {name:get_cluster_idx(name) for name in cluster_names}
cluster_titles      = [get_cluster_titles(name) for name in cluster_names]
cluster_titles      = {name:titles for name, titles in zip(cluster_names, cluster_titles)}

clusters_len        = [len(cluster_titles[cluster]) for cluster in cluster_names]
assert 0 not in clusters_len

clusters_dtm_rows = {name:dtm_lsa[idx, :]
                        for name, idx in zip(cluster_names, cluster_idx_list)}

centroids = make_centroids(cluster_names, cluster_idx_list, dtm_lsa)

##Calc silhouette and select best clusters
dist_dtm = cosine_distances(dtm_lsa)

clusters_silhouette  = {}
for key in clusters_dtm_rows.keys():
    clusters_silhouette[key] = calc_silhouette_for_cluster(key, cluster_idx_map, dist_dtm, data.shape[0])

clusters_ess = {}
for key in clusters_silhouette.keys():
    clusters_ess[key] = calc_cluster_ess(key, cluster_idx_map, centroids, dtm_lsa)

#Calc coherence score
clusters_coherence_score  = {}
for key in clusters_silhouette.keys():
    clusters_coherence_score[key] =  calc_cluster_coherence_score(key, clusters_len, clusters_silhouette, clusters_ess, data, dtm_lsa, avg_cluster_score)

sorted_coherence_scores = sorted(clusters_coherence_score.items(), key=lambda x: x[1], reverse = True)

# Gradual threshold adjustment
thresholds = np.linspace(start=max(sorted_coherence_scores, key=lambda x: x[1])[1], stop=min(sorted_coherence_scores, key=lambda x: x[1])[1], num=10)
clusters_by_threshold = {thresh: [idx for idx, score in sorted_coherence_scores if score > thresh] for thresh in thresholds}

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

thresholded_idx = [idx for idx, score in sorted_coherence_scores if score > score_thresh]


filtered_sorted_coherence_scores = {idx: score for idx, score in sorted_coherence_scores if idx not in thresholded_idx}


# Calculate the 90th percentile as the similarity threshold
centroid_dict = {cluster_id:centroids[cluster_id] for cluster_id in filtered_sorted_coherence_scores.keys()}
centroid_list = [centroids[cluster_id] for cluster_id in filtered_sorted_coherence_scores.keys()]
similarity_matrix = cosine_similarity(centroid_list)
similarity_threshold = np.percentile(np.sort(similarity_matrix.flatten()), 90)

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
merged_clusters_info = update_clusters_info(clusters_info, merged_centroids)

# Assuming 'merged_clusters_info' contains the updated indices and titles for merged clusters
updated_cluster_idx_map = {cluster_id: info['indices'] for cluster_id, info in merged_clusters_info.items()}
updated_clusters_dtm_rows = {cluster_id: dtm_lsa[info['indices'], :] for cluster_id, info in merged_clusters_info.items()}


updated_clusters_len_dict = {cluster_id: len(info['titles']) for cluster_id, info in merged_clusters_info.items()}

updated_clusters_silhouette = {
    cluster_id: calc_silhouette_for_cluster(cluster_id, updated_cluster_idx_map, dist_dtm, data.shape[0])
    for cluster_id in merged_clusters_info.keys()
}

# Assuming 'merged_centroids' contains centroids for merged clusters and 'centroids' for unmerged
combined_centroids = {**centroids, **merged_centroids}
updated_clusters_ess = {
    cluster_id: calc_cluster_ess(cluster_id, updated_cluster_idx_map, combined_centroids, dtm_lsa)
    for cluster_id in updated_cluster_idx_map.keys()
}

updated_avg_cluster_score = aggregate_avg_cluster_score(merged_clusters_info, avg_cluster_score)


updated_clusters_coherence_score = {
    cluster_id: calc_cluster_coherence_score(
        cluster_id,
        updated_clusters_len_dict,  # Pass the dictionary instead of the list
        updated_clusters_silhouette,
        updated_clusters_ess,
        data,
        dtm_lsa,
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
    
updated_clusters_dtm_rows = {cluster_id: dtm_lsa[info['indices'], :] for cluster_id, info in merged_clusters_info.items()}

updated_cluster_names = list(merged_clusters_info.keys())
updated_cluster_titles = {name: merged_clusters_info[name]["titles"] for name in updated_cluster_names}


# Sort by coherence
all_coherences = [(item[0], item[1]) for item in sorted_coherence_scores if item[0] in thresholded_idx] + [(item[0], item[1]) for item in update_sorted_coherence_scores if item[0] in updated_thresholded_idx]
sorted_all_coherences = sorted(all_coherences, key=lambda x: x[1], reverse = True)


# Use merged centroids for group definitions
group_centroids = {key:merged_centroids[key] for key in [item[0] for item in sorted_all_coherences]}

# Updated lambda function to calculate dot product using new centroids
cmp_arts_centroid = lambda x: np.dot(updated_clusters_dtm_rows[x], group_centroids[x])

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
sample_dtm = {cluster: updated_clusters_dtm_rows[cluster][resampled_idx[cluster], :] for cluster in resampled_idx.keys()}

article_top_concepts = get_article_top_concepts(dtm_lsa)


#Get exemplar
get_exemplar_idx = lambda x: np.where(art_dot_centroid[x] == art_dot_centroid[x].max())
cluster_exemplars = {cluster: updated_cluster_titles[cluster][get_exemplar_idx(cluster)[0][0]] for cluster in group_idx}


##Get most relevant from group
avg_centroid = dtm_lsa.mean(axis=0)
centroids_to_avg = {cluster_id: np.dot(merged_centroids[cluster_id], avg_centroid) for cluster_id in group_centroids.keys()}

cmp_cluster_to_avg = lambda centroid: np.dot(updated_clusters_dtm_rows[centroid], centroids_to_avg[centroid])
cluster_cmp_avg = {centroid:cmp_cluster_to_avg(centroid) for centroid in centroids_to_avg.keys()}

get_relevant_idx = lambda x: np.where(cluster_cmp_avg[x] == cluster_cmp_avg[x].max())
cluster_most_relevant = {cluster:updated_cluster_titles[cluster][get_relevant_idx(cluster)[0][0]]
                        for cluster in cluster_cmp_avg.keys()}


#Get cluster top, infreq terms
centroids_top_terms = {}
all_infreq_top_centroid_terms = Counter()
for cluster_id in group_centroids.keys():
    top_terms = get_cluster_top_terms(cluster_id, merged_centroids, lsa, terms)[:50]
    centroids_top_terms[cluster_id] = top_terms
    all_infreq_top_centroid_terms.update(top_terms)

max_cnt = sorted(all_infreq_top_centroid_terms.items(), key= lambda x: x[1], reverse = True)[0][1]
max_freq = max_cnt//2 if max_cnt > 7 else max_cnt

centroids_infreq_top_terms = {}
for cluster_id in group_centroids.keys():
    top_terms = get_cluster_top_terms(cluster_id, group_centroids, lsa, terms)[:50]
    centroids_infreq_top_terms[cluster_id] = [term for term in top_terms if all_infreq_top_centroid_terms[term] < max_freq][:20]



###
#TODO:
#--Use those articles to create a train/test sample
#--preprocess those tagged articles and decompose - check those concepts
#--Train a DTC using those samples
#--Tag all articles into those groups
#--Remove the unwanted groups [/articles]
#--Repeat ALL 1-2 more times
#
##

for g in group_idx:
    print(g)
    pprint(get_cluster_top_terms(g, merged_centroids, lsa, terms)[:20])
    pprint(sample_titles[g])
    input()


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
    data_idx = [data.loc[data.Title == title].index[0] for title in titles]
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
    pickle.dump([group_centroids, merged_clusters_info, updated_clusters_dtm_rows, updated_cluster_names, updated_cluster_titles, sorted_all_coherences], p)

