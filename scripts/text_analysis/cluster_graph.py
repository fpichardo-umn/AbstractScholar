# -*- coding: utf-8 -*-
"""
STEP 02: Cluster Graph

This script performs clustering analysis on a dataset of academic articles based on their abstracts. It utilizes Latent Semantic Analysis (LSA) to preprocess and vectorize the text data, and then applies the k-means clustering algorithm to group the articles into clusters. The optimal number of clusters (k) is determined by selecting the value that maximizes the Bayesian Information Criterion (BIC) (other similarity measures available).

To analyze the similarity between clusters, the script calculates pairwise edit distances between cluster assignments. A network graph is constructed using the NetworkX package, where nodes represent documents and edges represent similarities above a specified threshold. The relationships between clusters are visualized in this graph. The resulting clusters are sorted by size and their respective article titles are written to a text file.

The script outputs two pickle files:
1. A file containing the clusters and their average clustering scores
2. A file containing the network graph

And two text files:
1. A file containing the titles of articles in each cluster
2. An updated dataset with cluster assignments

Description of Method:
1. **Preprocessing**: Previous script (step 01) which preprocesses and vectorizes the text data using Latent Semantic Analysis (LSA).
2. **Clustering**: Multiple values of k are tested using k-means clustering, and the Bayesian Information Criterion (BIC) (or another similarity measure) is used to select the optimal number of clusters.
3. **Robustness**: To ensure robust clustering results, the k-means clustering process is repeated multiple times. Each document's cluster assignments from different runs are concatenated into a string of digits.
4. **Edit Distance Calculation**: The pairwise edit distances between these concatenated strings are computed, representing the similarity between the clustering results for each pair of documents.
5. **Graph Construction**: A network graph is created where each document is a node and edges represent similarities (edit distances) below a specified threshold.
6. **Cluster Identification**: Graph theory algorithms are used to identify connected components in the graph, corresponding to document clusters. Subgraphs with more than one node are considered clusters.
7. **Output**: Clusters are sorted by size, and their article titles are written to a text file. Clustering results and the network graph are saved as pickle files.

Created on Sat Apr 14 15:19:05 2018

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
import Levenshtein
import networkx as nx
from difflib import SequenceMatcher
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score


def compute_bic(kmeans, X):
    """
    Computes the Bayesian Information Criterion (BIC) metric for a given set of clusters.

    Args:
        kmeans (list): List of clustering objects from scikit-learn.
        X (ndarray): Multidimensional numpy array of data points.

    Returns:
        float: The BIC value. Higher values indicate better clustering.
    """
    
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    
    # number of clusters
    m = kmeans.n_clusters
    
    # size of the clusters
    n = np.bincount(labels)
    
    # size of data set
    N, d = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d + 1)

    BIC = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                  ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

    return BIC


def kmeans_test_grouping(data_vector, ncluster_max, ncluster_min, kmeans_init='random', n_init=30, sim='bic'):
    """
    Performs k-means clustering on the given data vector and returns the optimal number of clusters based on the specified similarity measure.

    Args:
        data_vector (array-like): The input data vector to be clustered.
        ncluster_max (int): The maximum number of clusters to consider.
        ncluster_min (int): The minimum number of clusters to consider.
        kmeans_init (str, optional): The method to initialize the centroids. Defaults to 'random'.
        n_init (int, optional): The number of times the k-means algorithm will be run with different centroid seeds. Defaults to 30.
        sim (str, optional): The similarity measure to determine the optimal number of clusters. Can be 'bic' (Bayesian Information Criterion), 'sil' (Silhouette Score), or 'cal' (Calinski-Harabasz Score). Defaults to 'bic'.

    Returns:
        tuple: A tuple containing two lists - ns and scores. 
            - ns (list): The list of number of clusters considered.
            - scores (list): The list of similarity scores corresponding to each number of clusters.
    """
    
    scores = []
    ns = []
    for k in range(ncluster_min, ncluster_max, 5):
        clf = KMeans(k, init=kmeans_init, n_init=n_init).fit(data_vector)

        if sim.lower() == 'bic':
            scores.append(compute_bic(clf, data_vector))
        elif sim.lower() == 'sil':
            scores.append(silhouette_score(data_vector, clf.labels_))
        elif sim.lower() == 'cal':
            scores.append(calinski_harabasz_score(data_vector, clf.labels_))
        else:
            raise ValueError("Invalid similarity measure. Please choose one of 'bic', 'sil', or 'cal'.")
        ns.append(k)

    print(ns[scores.index(max(scores))])

    return ns, scores


def test_groups(data_vector, iterations=10, max_k=30, min_k=2, kmeans_init='random', n_init=30, sim='bic'):
    """
    Test different groupings using k-means clustering algorithm.

    Args:
        data_vector (list): The data vector to be clustered.
        iterations (int, optional): The number of times to run the clustering algorithm. Defaults to 10.
        max_k (int, optional): The maximum number of clusters to test. Defaults to 30.
        min_k (int, optional): The minimum number of clusters to test. Defaults to 2.
        kmeans_init (str, optional): The method to initialize the k-means algorithm. Defaults to 'random'.
        n_init (int, optional): The number of times the k-means algorithm will be run with different centroid seeds. Defaults to 30.
        sim (str, optional): The similarity measure to use for selecting the best number of clusters. Defaults to 'bic'.

    Returns:
        list: A list of the best number of clusters for each run of the clustering algorithm.
    """
    
    best_ns = []
    for i in range(iterations):
        ns, scores = kmeans_test_grouping(data_vector, max_k, min_k,
                                          kmeans_init=kmeans_init, n_init=n_init, sim=sim)
        n = ns[scores.index(max(scores))]

        best_ns += [n]

    return best_ns


def determine_best_k(doc_term_mat_xfm, group_iterations, kmeans_max_k, kmeans_init, kmeans_sim):
    """
    Determine the best number of clusters using multiple iterations.

    Args:
        doc_term_mat_xfm (numpy.ndarray): The transformed document-term matrix.
        group_iterations (int): The number of iterations to perform of the clustering algorithm.
        kmeans_max_k (int): The maximum number of clusters to consider.
        kmeans_init (str): The initialization method for K-means clustering.
        kmeans_sim (str): The similarity measure to use for K-means clustering.

    Returns:
        int: The best number of clusters determined by averaging the results of multiple iterations.
    """
    best_ns = np.array(test_groups(doc_term_mat_xfm, iterations=group_iterations, min_k=2, max_k=min(kmeans_max_k, doc_term_mat_xfm.shape[0]), kmeans_init=kmeans_init, sim=kmeans_sim))
    average_k = best_ns.sum() / best_ns.size
    
    print(f'Best k: {average_k}')
    return int(round(average_k))


def group_abstracts_ensemble(data, doc_term_mat_xfm, best_k, num_groupings, kmeans_init, k_n_init):
    to_printable = np.vectorize(lambda idx: string.printable[idx % len(string.printable)])
    data['group_strs'] = ''

    # Estimate initial eps for DBSCAN
    initial_eps = estimate_dbscan_eps(doc_term_mat_xfm)*.7

    for a in range(num_groupings):
        if a % 5 == 0:
            print(f"Iteration {a}/{num_groupings}")

        # K-means (as before)
        kmeans = KMeans(n_clusters=best_k, init=kmeans_init, n_init=k_n_init).fit(doc_term_mat_xfm)
        
        # DBSCAN with varying parameters
        min_samples = (a % 5) + 2  # Vary min_samples between 2 and 6
        eps = initial_eps * (1 + 0.1 * (a // 5))  # Increase eps by 10% every 5 iterations
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(doc_term_mat_xfm)
        
        # Agglomerative Clustering with varying parameters
        linkage = ['ward', 'complete', 'average', 'single'][a % 4]
        n_clusters = max(2, int(best_k * (0.8 + 0.4 * (a / num_groupings))))  # Vary n_clusters between 80% and 120% of best_k
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(doc_term_mat_xfm)

        # Process labels
        kmeans_labels = to_printable(kmeans.labels_)
        
        dbscan_labels = dbscan.labels_
        dbscan_labels[dbscan_labels == -1] = max(dbscan_labels) + 1
        dbscan_labels = to_printable(dbscan_labels)
        
        agg_labels = to_printable(agg.labels_)

        combined_labels = np.core.defchararray.add(np.core.defchararray.add(kmeans_labels, dbscan_labels), agg_labels)

        data['group_strs'] = data['group_strs'].values + combined_labels

    return data

def estimate_dbscan_eps(X, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    return np.median(distances)  # Use median as a robust estimate


def calc_edist(a, b):
    """Calculate the normalized Levenshtein distance between two strings."""
    if not a and not b:  # Handle the case where both strings are empty
        return 0.0
    levenshtein_distance = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return levenshtein_distance / max_len


def gen_edist_matrix(data):
    """
    Generates an edit distance matrix based on the 'group_strs' column of the given data using the normalized Levenshtein distance.

    Args:
        data (pandas.DataFrame): The input data containing the 'group_strs' column.

    Returns:
        numpy.ndarray: The edit distance matrix.
    """
    num_arts = len(data)
    edist = np.zeros((num_arts, num_arts))
    group_strs = data['group_strs'].tolist()  # Assuming 'group_strs' is the column name

    for i in range(num_arts):
        if i % 100 == 0:
            print(i)  # Progress indication
        for j in range(i + 1, num_arts):
            edist[i, j] = calc_edist(group_strs[i], group_strs[j])

    # Use the computed upper triangle to fill the lower triangle
    lower_tri = np.tril_indices_from(edist, -1)
    edist[lower_tri] = edist.T[lower_tri]

    # Set the diagonal values to 1.0
    np.fill_diagonal(edist, 0)

    return edist


def cluster_graphs(edist, max_edist):
    """
    Cluster graphs based on the edit distance matrix and maximum edit distance threshold.

    Args:
        edist (numpy.ndarray): The edit distance matrix.
        max_edist (float): The maximum edit distance threshold.

    Returns:
        tuple: A tuple containing the following:
            - list: A list of clusters, where each cluster is a subgraph of the original graph.
            - list: A list of solo nodes, where each solo node is a subgraph of the original graph.
            - networkx.Graph: The graph with only strong connections.
    """
    
    G = nx.from_numpy_matrix(edist)
    
    strong_connection = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < max_edist]
    G_strong = nx.Graph()
    G_strong.add_edges_from(strong_connection)
    
    all_components = list(nx.connected_components(G_strong))
    
    nodes_in_cluster_list = []
    clusters = []
    solo = []
    
    for nodes in all_components:
        sub_graph = G_strong.subgraph(nodes)
        
        nodes_in_cluster_list += list(nodes)
        
        if len(sub_graph.nodes()) > 1:
            clusters.append(sub_graph)
        else:
            solo.append(sub_graph)
    
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    
    # Add remaining nodes to the solo list
    solo += [node for node in list(G.nodes) if node not in nodes_in_cluster_list]
    
    return clusters, solo, G_strong


def assign_clusters_to_data(data, clusters, solo):
    """
    Assigns cluster labels to the data DataFrame.

    Args:
        data (DataFrame): The DataFrame containing the data.
        clusters (list): A list of clusters, where each cluster is represented as a graph.
        solo (list): A list of solo clusters, where each solo cluster is represented as a graph.

    Returns:
        DataFrame: The data DataFrame with cluster labels assigned.
    """
    
    data['cluster'] = ''
    data.reset_index(inplace=True)
    
    for cluster_idx, cluster in enumerate(clusters):
        article_idx = list(cluster.nodes())
        data.loc[article_idx, 'cluster'] = cluster_idx
    
    for solo_node in solo:
        data.loc[solo_node, 'cluster'] = 999  # Use 999 for solo clusters


def write_cluster_titles(data, clusters, cluster_titles_txt):
    """Write cluster titles to a file."""
    cluster_sort = list(range(len(clusters))) + [999]
    
    with open(cluster_titles_txt, 'w', encoding='iso-8859-1') as f:
        for cluster in cluster_sort:
            cluster_name = cluster if cluster != 999 else 'solo'
            f.write('Cluster {}:\n'.format(cluster_name))
            cluster_values = data.loc[data.cluster == cluster, 'title'].values
            for val in cluster_values:
                f.write(val)
                f.write('\n')
            f.write('\n\n')
    
    print('Num clusters: {}\nNum solo articles: {}'.format(len(clusters), len(solo)))


def adaptive_threshold(edist_matrix, percentile=1):
    # Flatten the matrix and filter out 1s and 0s
    filtered_edist_matrix = edist_matrix[(edist_matrix != 1) & (edist_matrix != 0)]
    
    # Check if the filtered array is not empty to avoid errors
    if filtered_edist_matrix.size == 0:
        raise ValueError("No valid elements left after filtering out 1s and 0s.")
    
    # Use a percentile of the edit distances as the threshold
    return np.percentile(filtered_edist_matrix, percentile)

####
##    START
####

# Load preprocessed data
config = load_user_config()
data = load_data(config)

# Get the configuration parameters
cluster_metic = config.get('cluster_metic', 'euclidean')
kmeans_init = config.get('kmeans_init', 'k-means++')
kmeans_sim = config.get('kmeans_sim', 'bic')
kmeans_max_k = int(config.get('kmeans_max_k', 50))
ncluster_max = int(config.get('ncluster_max', 60))
num_groupings = int(config.get('num_groupings', 35))
k_n_init = int(config.get('k_n_init', 35))
group_iterations = int(config.get('group_iterations', 10))
max_edist = float(config.get('max_edist', 0.2))
preprocess_pickle_filename = normalize_path(config.get('preprocess_pickle', './data/text_analysis/large_files/preprocessed_abstracts.pickle'))
edist_abstract_pickle_filename = normalize_path(config.get('edist_abstract_pickle', './data/text_analysis/large_files/edist.pickle'))
cluster_titles_txt = normalize_path(config.get('cluster_titles_txt', './data/text_analysis/clustered_data.txt'))
clustered_data_csv = normalize_path(config.get('clustered_data_csv', './data/text_analysis/clustered_data.csv'))
clusters_pickle_filename = normalize_path(config.get('clusters_pickle', './data/text_analysis/large_files/clusters.pickle'))

# Load preprocessed abstract data
latent_sa, doc_term_mat_xfm, terms = load_from_pickle(preprocess_pickle_filename)

# Get best k
best_k = determine_best_k(doc_term_mat_xfm, group_iterations, kmeans_max_k, kmeans_init, kmeans_sim)

# Group abstracts with kmeans multiple times
data = group_abstracts_ensemble(data, doc_term_mat_xfm, best_k, num_groupings, kmeans_init, k_n_init)

# Calc and save edit distance matrix
edist = gen_edist_matrix(data)
save_to_pickle(edist, edist_abstract_pickle_filename)

# Cluster Graphs
max_edist = adaptive_threshold(edist)
clusters, solo, G_strong = cluster_graphs(edist, max_edist)

# Assign clusters to data
assign_clusters_to_data(data, clusters, solo)

# Write cluster titles to a file
write_cluster_titles(data, clusters, cluster_titles_txt)

# Calculate the average clustering score for each cluster
avg_cluster_score = {cluster_idx: nx.algorithms.average_clustering(G_strong.subgraph(cluster))
                     for cluster_idx, cluster in enumerate(clusters)}

# Save clustering data
data.to_csv(clustered_data_csv, sep = ',', encoding='iso-8859-1', index = False)
save_to_pickle([clusters, avg_cluster_score, G_strong], clusters_pickle_filename)