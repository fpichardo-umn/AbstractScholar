# -*- coding: utf-8 -*-
"""
STEP 04: Train Article Group Classifier

This script performs the following tasks:
1. Loads preprocessed data and cluster information
2. Labels articles based on cluster relevance
3. Trains an ElasticNet classifier to predict article relevance
4. Predicts relevance for unlabeled articles
5. Categorizes articles as Relevant, Irrelevant, or Borderline
6. Samples articles from each category for review

The process involves:
- Using cluster information to label a subset of articles
- Training a classifier on labeled data
- Applying the classifier to unlabeled data
- Using a combination of ML predictions, cluster coherence, and distances to centroids for final categorization

Output:
- Prints performance metrics of the classifier
- Displays samples of articles from each category (Relevant, Borderline, Irrelevant)

Requirements:
- Preprocessed document-term matrix and LSA components
- Cluster information from previous steps
- Configuration file with necessary parameters

Created on Sat Apr 14 15:47:29 2018
@author: Felix Pichardo (original)
"""

import sys
import os
import os.path as op
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_text import *

def normalize_values(coherences):
    """Normalize coherence values."""
    max_value = max(coherences.values())
    return {key: value / max_value for key, value in coherences.items()}

def load_data_and_clusters(config):
    """Load data, preprocessed abstracts, and cluster information."""
    data = load_data(config, clustered_data=True)
    data.reset_index(drop=True, inplace=True)
    
    preprocess_pickle_filename = normalize_path(config.get('preprocess_pickle', './data/text_analysis/large_files/preprocessed_abstracts.pickle'))
    latent_sa, doc_term_mat_xfm, terms = load_from_pickle(preprocess_pickle_filename)
    
    cluster_pickle = op.join('data', 'cluster_abstracts.pickle')
    with open(cluster_pickle, 'rb') as p:
        group_centroids, merged_clusters_info, updated_clusters_dtm_rows, \
            updated_cluster_names, updated_cluster_titles, sorted_all_coherences = \
                pickle.load(p)
    
    return data, latent_sa, doc_term_mat_xfm, terms, group_centroids, merged_clusters_info, sorted_all_coherences

def label_articles(data, merged_clusters_info, relevant_clusters, irrelevant_clusters):
    """Label articles based on cluster membership."""
    data['label'] = None
    for cluster in relevant_clusters:
        data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 1
    for cluster in irrelevant_clusters:
        data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 0
    return data

def prepare_features(data, doc_term_mat_xfm, group_centroids, normalized_coherences):
    """Prepare features for the classifier."""
    for cluster_id, centroid in group_centroids.items():
        indices = merged_clusters_info[cluster_id]['indices']
        data.loc[indices, 'distance_to_centroid'] = cosine_distances(doc_term_mat_xfm[indices], centroid.reshape(1, -1)).flatten()
        data.loc[indices, 'coherence_weight'] = normalized_coherences[cluster_id]
    
    labeled_data = data.dropna(subset=['label'])
    labeled_features = doc_term_mat_xfm[labeled_data.index]
    features = np.hstack([labeled_features, labeled_data[['distance_to_centroid', 'coherence_weight']].values])
    labels = labeled_data['label']
    
    return features, labels

def train_and_evaluate_classifier(features, labels):
    """Train and evaluate the ElasticNet classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    enet = ElasticNetCV(cv=5)
    enet.fit(X_train, y_train)

    predictions = enet.predict(X_test)
    predictions_binary = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions_binary)
    precision = precision_score(y_test, predictions_binary)
    recall = recall_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return enet

def predict_unlabeled_articles(data, doc_term_mat_xfm, group_centroids, normalized_coherences, enet):
    """Predict relevance for unlabeled articles."""
    # Create an explicit copy of the unlabeled data
    unlabeled_data = data[data['label'].isna()].copy()

    # Initialize columns using .loc to avoid warnings
    unlabeled_data.loc[:, 'nearest_centroid'] = None
    unlabeled_data.loc[:, 'distance_to_nearest_centroid'] = np.inf

    for cluster_id, centroid in group_centroids.items():
        distances = cosine_distances(doc_term_mat_xfm[unlabeled_data.index], centroid.reshape(1, -1)).flatten()
        mask = distances < unlabeled_data['distance_to_nearest_centroid']
        unlabeled_data.loc[mask, 'nearest_centroid'] = cluster_id
        unlabeled_data.loc[mask, 'distance_to_nearest_centroid'] = distances[mask]

    unlabeled_data.loc[:, 'coherence_weight'] = unlabeled_data['nearest_centroid'].map(normalized_coherences).fillna(0)

    unlabeled_features = np.hstack([
        doc_term_mat_xfm[unlabeled_data.index],
        unlabeled_data[['distance_to_nearest_centroid', 'coherence_weight']].values
    ])

    unlabeled_predictions = enet.predict(unlabeled_features)
    unlabeled_data.loc[:, 'predicted_1_prob'] = 1 / (1 + np.exp(-unlabeled_predictions))
    unlabeled_data.loc[:, 'predicted_label'] = (unlabeled_predictions > 0.5).astype(int)

    return unlabeled_data

def categorize_articles(data, doc_term_mat_xfm, group_centroids, normalized_coherences):
    """Categorize articles as Relevant, Irrelevant, or Borderline."""
    # Create an explicit copy
    unlabeled_data = data.copy()

    min_max_normalize = lambda x: (x - x.min()) / (x.max() - x.min())

    irrelevant_centroids = {cid: group_centroids[cid] for cid in relevant_clusters}

    unlabeled_data.loc[:, 'sim_to_nearest_irrelevant_centroid'] = -np.inf
    unlabeled_data.loc[:, 'nearest_irrelevant_centroid_coherence'] = 0

    for cluster_id, centroid in irrelevant_centroids.items():
        sims = cosine_similarity(doc_term_mat_xfm[unlabeled_data.index], centroid.reshape(1, -1)).flatten()
        for idx, sim in zip(unlabeled_data.index, sims):
            if sim > unlabeled_data.at[idx, 'sim_to_nearest_irrelevant_centroid']:
                unlabeled_data.at[idx, 'sim_to_nearest_irrelevant_centroid'] = sim
                unlabeled_data.at[idx, 'nearest_irrelevant_centroid_coherence'] = normalized_coherences.get(cluster_id, 0)

    unlabeled_data['final_score'] = min_max_normalize(unlabeled_data['predicted_1_prob'] - (unlabeled_data['sim_to_nearest_irrelevant_centroid'] * unlabeled_data['nearest_irrelevant_centroid_coherence'])**(1/8))

    mean_score = unlabeled_data['final_score'].mean()
    std_score = unlabeled_data['final_score'].std()
    lower_threshold = mean_score - std_score
    upper_threshold = mean_score + std_score

    kmeans = KMeans(n_clusters=3, random_state=42).fit(unlabeled_data[['final_score']])
    centroids = kmeans.cluster_centers_.flatten()
    sorted_centroids = np.sort(centroids)

    final_lower_threshold = (lower_threshold + sorted_centroids[0]) / 2
    final_upper_threshold = (upper_threshold + sorted_centroids[-1]) / 2

    unlabeled_data['category'] = np.where(unlabeled_data['final_score'] <= final_lower_threshold, 'Irrelevant',
                                          np.where(unlabeled_data['final_score'] >= final_upper_threshold, 'Relevant', 'Borderline'))

    return unlabeled_data

def sample_articles(unlabeled_data):
    """Sample articles from each category."""
    relevant_sample = unlabeled_data[unlabeled_data['category'] == 'Relevant'][['title', 'abstract']].sample(5)
    borderline_sample = unlabeled_data[unlabeled_data['category'] == 'Borderline'][['title', 'abstract']].sample(5)
    irrelevant_sample = unlabeled_data[unlabeled_data['category'] == 'Irrelevant'][['title', 'abstract']].sample(5)

    print("Relevant Articles Sample:\n", relevant_sample)
    print("\nBorderline Articles Sample:\n", borderline_sample)
    print("\nIrrelevant Articles Sample:\n", irrelevant_sample)

def export_categorized_data(categorized_data, output_file='data/categorized_articles.csv'):
    """
    Export categorized data to a CSV file for user editing.
    
    Args:
    categorized_data (pd.DataFrame): The DataFrame containing categorized article data.
    output_file (str): The name of the output CSV file.
    
    Returns:
    None
    """
    # Select the required columns
    columns_to_export = [
        'orig_index', 'title', 'year', 'journal', 'issn', 'volume', 'issue', 
        'pages', 'authors', 'language', 'abstract', 'doi', 'keywords', 
        'orig_abstract', 'label', 'final_score', 'category'
    ]
    
    # Create a new DataFrame with selected columns
    export_df = categorized_data[columns_to_export].copy()
    
    # Sort the DataFrame by final_score in ascending order
    export_df = export_df.sort_values('final_score', ascending=True)
    
    # Add a new column for user judgment
    export_df['user_judgment'] = ''
    
    # Reorder columns to put user_judgment at the end
    columns_order = columns_to_export + ['user_judgment']
    export_df = export_df[columns_order]
    
    # Export to CSV
    export_df.to_csv(output_file, index=False)
    
    print(f"Data exported to {output_file}")

config = load_user_config()
data, latent_sa, doc_term_mat_xfm, terms, group_centroids, merged_clusters_info, sorted_all_coherences = load_data_and_clusters(config)

relevant_clusters = [1, 'merged_3_24', 'merged_3_19']
irrelevant_clusters = [2, 10, 4]

data = label_articles(data, merged_clusters_info, relevant_clusters, irrelevant_clusters)

normalized_coherences = normalize_values(dict(sorted_all_coherences))

features, labels = prepare_features(data, doc_term_mat_xfm, group_centroids, normalized_coherences)

enet = train_and_evaluate_classifier(features, labels)

unlabeled_data = predict_unlabeled_articles(data, doc_term_mat_xfm, group_centroids, normalized_coherences, enet)

categorized_data = categorize_articles(unlabeled_data, doc_term_mat_xfm, group_centroids, normalized_coherences)

export_categorized_data(categorized_data, output_file = normalize_path(config.get('categorized_output_file', './data/text_analysis/categorized_articles.csv')))

sample_articles(categorized_data)