# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:19:05 2018

@author: Sulla

The script performs clustering analysis on a dataset of academic articles based on their abstracts. It uses Latent Semantic Analysis (LSA) to preprocess and vectorize the text data, and then applies k-means clustering algorithm to group the articles into clusters. The number of clusters is determined by selecting the value of k that optimizes the Bayesian information criterion (BIC).

The script uses the edit distance metric to calculate the similarity between clusters, and constructs a network graph to visualize the relationships between the clusters. The graph is generated using the NetworkX package, and strong connections between clusters are defined as those with a similarity above a specified threshold. The resulting clusters are sorted by size and written to a text file along with their respective article titles.

The script outputs two pickle files: one containing the clusters and their average clustering scores, and another containing the network graph.

Description of Method:
The script uses k-means clustering to group the documents into clusters based on their similarity in the reduced topic space. The script tries multiple values of k and uses the Bayesian Information Criterion (BIC) to determine the optimal number of clusters.

To ensure robustness of the clustering results, the script repeats the k-means clustering process multiple times and concatenates the resulting cluster assignments for each document. This produces a string of digits, where each digit corresponds to a different cluster assignment from a different run of k-means clustering.

The script then calculates the pairwise edit distances between these strings of digits, which represent the similarity between the clustering results for each pair of documents. This distance matrix is then used to construct a graph, where each document is a node and the edges represent the similarity between pairs of documents.

Finally, the script applies graph theory algorithms to identify the connected components of the graph, which correspond to the document clusters. The script identifies clusters as subgraphs with more than one node, and assigns each document to the cluster corresponding to its connected component.
"""

import string
import pickle
import os.path as op
import numpy as np
import pandas as pd
import networkx as nx
from difflib import SequenceMatcher
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score


def similar(a, b):
    "Return ratio of exact positional matches between strs a and b"
    return SequenceMatcher(None, a, b).ratio()


def compute_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
        Higher is better
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return BIC


def kmeans_test_grouping(data_vector, ncluster_max, ncluster_min, kmeans_init = 'random', n_init = 30, sim = 'bic'):

    scores = []
    ns = []
    for k in range(ncluster_min, ncluster_max, 5):
        clf = KMeans(k, init = kmeans_init, n_init = n_init).fit(data_vector)

        if sim.lower() == 'bic':
            scores.append(compute_bic(clf, data_vector))
        elif sim.lower() == 'sil':
            scores.append(silhouette_score(data_vector, clf.labels_))
        else:
            scores.append(calinski_harabasz_score(data_vector, clf.labels_))
        ns.append(k)

    print(ns[scores.index(max(scores))])

    return ns, scores



def get_article_top_concepts(docs, concepts_transformed_data):

    art_top_concepts = {art:[] for art in range(len(docs))}
    top_num = 5
    for i in range(len(concepts_transformed_data)):
       top_topics = np.argsort(concepts_transformed_data[i,:])[::-1][:top_num]
       top_topics_str = ' '.join(str(t) for t in top_topics)
       art_top_concepts[i] = [int(top) for top in top_topics_str.split()]

    return art_top_concepts

def get_terms_for_concept(terms, components, concept_idx, num_terms = 20):

    comp = components[concept_idx]
    terms_in_comp = zip(terms, comp)
    sorted_terms = sorted(terms_in_comp, key = lambda x: x[1], reverse = True) [:num_terms]
    sorted_terms = [term[0] for term in sorted_terms]

    return sorted_terms


def get_top_words_art(art_idx_list, terms, components, concepts_per_art = 3, terms_per_concept = 10):

    if not 'article_top_concepts' in dir():
        article_top_concepts = get_article_top_concepts(docs, dtm_lsa)

    concepts = []
    for art in art_idx_list:
        concepts.extend(article_top_concepts[art][:concepts_per_art])

    concepts = list(set(concepts))

    top_terms = []
    for concept in concepts:
        top_terms.append(get_terms_for_concept(terms, components, concept, terms_per_concept))

    return top_terms


def test_groups(data_vector, times = 10, max_k = 30, min_k = 2, kmeans_init = 'random', n_init = 30, sim = 'bic'):

    gs = []
    for i in range(times):
        ns, scores = kmeans_test_grouping(data_vector, max_k, min_k,
                                          kmeans_init = kmeans_init, n_init = n_init, sim = sim)
        n = ns[scores.index(max(scores))]

        gs += [n]

    return gs


def pickle_data(pickle_fname, data = None, kind = 'save'):
    """Save or load pickle file"""

    if kind == 'save':
        with open(pickle_fname, 'wb') as f:
            pickle.dump(data, f)
    elif kind == 'load':
        with open(pickle_fname, 'rb') as f:
            return pickle.load(f)



cluster_metic = 'euclidean'
ncluster_max = 60
kmeans_init = 'k-means++'
num_groupings = 35
k_n_init = 35

pulled_data = 'pulled_abs.txt'
data = pd.read_csv(op.join('data', pulled_data), sep = '\t', encoding='iso-8859-1', index_col = None)
#data = data.loc[data[data.abstract.apply(str).apply(len) > 50].index]
data['abstract'] = data['abstract'].fillna('')
data = data[data.abstract.apply(len) > 50]
data = data.drop_duplicates('Title').reset_index(drop = True)

preprocess_pickle = op.join('data', 'preprocessed_abstracts.pickle')
with open(preprocess_pickle, 'rb') as p:
    lsa, dtm_lsa, terms = pickle.load(p)


#Show Concept terms
get_terms_for_concept(terms, lsa.components_, 0)

###
#   Get best k
###
gs = test_groups(dtm_lsa, times = 5, min_k=2, max_k=min(50, dtm_lsa.shape[0]), kmeans_init = 'k-means++', sim = 'bic')
gs = np.array(gs)
print(gs.sum()/gs.size)
k = (gs.sum()/gs.size).round(0).astype(int)

###
#   Group abstracts with kmeans multiple times
###

to_printable = np.vectorize(lambda idx: string.printable[idx])

data['gstrs'] = ''
for a in range(num_groupings):
    if a % 5 == 0:
        print(a)
    clf = KMeans(k, init = kmeans_init, n_init = k_n_init).fit(dtm_lsa)

    groups = pd.DataFrame(np.zeros_like(data.Title), columns = ['group'])
    groups.group = to_printable(clf.labels_)

    data_groups = groups.reset_index(drop = True)

    data['gstrs'] = data['gstrs'].str.cat(groups.group)

###
#   Calculate edit distance matrix
###
art_num = data.shape[0]
edist = np.zeros((art_num, art_num))
edist[np.diag_indices_from(edist)] = 1.0

for art in range(art_num):
    if art % 100 == 0:
        print(art)
    if art + 1 >= art_num:
        break
    for comp in range(art + 1, art_num):
        edist[art][comp] = similar(data.iloc[art].gstrs, data.iloc[comp].gstrs)

lower_tri = np.tril_indices_from(edist)
edist[lower_tri] = edist.T[lower_tri]

edist_abs_pickle = 'edist_abstracts.pickle'
pickle_data(edist_abs_pickle, edist)

###
#   Cluster Graphs
###

min_dist = .8
G = nx.from_numpy_matrix(edist)
pos = nx.spring_layout(G)

strong_connection = [(u,v) for (u,v,d) in G.edges(data = True) if d['weight'] > min_dist]

nx.draw_networkx_edges(G, pos, edgelist = strong_connection, width=1)

G_strong = nx.Graph(strong_connection)
all_components = list(nx.components.connected_components(G_strong))

# Create subgraphs for each connected component
components = []
for nodes in all_components:
    sub_graph = G_strong.subgraph(nodes)
    components.append(sub_graph)

clusters = [comp for comp in components if len(comp.nodes()) > 1]
clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
solo = [comp for comp in components if len(comp.nodes()) == 1]

components = []
G_components = nx.components.connected_components(G_strong)
for comp in range(nx.components.connected.number_connected_components(G_strong)):
    sub_graph = next(G_components)
    # print(len(sub_graph), sub_graph)
    components.append(sub_graph)

clusters    = [comp for comp in components if len(comp) > 1]
clusters    = sorted(clusters, key=lambda x: len(x), reverse = True)
solo        = [comp for comp in components if len(comp) == 1]

data['cluster'] = ''
for cluster in range(len(clusters)):
    article_idx = list(clusters[cluster])
    data.loc[article_idx, 'cluster'] = cluster

for cluster in range(len(solo)):
    article_idx = list(solo[cluster])
    data.loc[article_idx, 'cluster'] = 999 #Use 999 for solo clusters

cluster_sort = list(range(len(clusters))) + [999]
with open('cluster_titles.txt', 'w', encoding='iso-8859-1') as f:
    for cluster in cluster_sort:
        cluster_name = cluster if cluster != 999 else 'solo'
        f.write('Cluster {}:\n'.format(cluster_name))
        cluster_values = data.loc[data.cluster == cluster, 'Title'].values
        for val in cluster_values:
            f.write(val)
            f.write('\n')
        f.write('\n\n')

print('Num clusters: {}\nNum solo articles:{}'.format(len(clusters), len(solo)))

avg_cluster_score = {cluster_idx: nx.algorithms.average_clustering(G_strong.subgraph(cluster))
                     for cluster_idx, cluster in enumerate(clusters)}


data_for_edist = op.join('data', 'clustered_data.txt')
data.to_csv(data_for_edist, sep = '\t', encoding='iso-8859-1', index = False)

clusters_pickle = op.join('data', 'clusters.pickle')
pickle_data(clusters_pickle, [clusters, avg_cluster_score, G_strong])
