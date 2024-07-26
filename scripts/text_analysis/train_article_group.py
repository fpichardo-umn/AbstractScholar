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
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

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

def label_articles(data, merged_clusters_info, relevant_clusters, irrelevant_clusters, borderline_clusters):
    data['label'] = 'unknown'  # Default label
    for cluster in relevant_clusters:
        data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 'relevant'
    for cluster in irrelevant_clusters:
        data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 'irrelevant'
    for cluster in borderline_clusters:
        data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 'borderline'
    return data

def prepare_features(data, doc_term_mat_xfm, group_centroids, normalized_coherences):
    """Prepare features for the classifier."""
    for cluster_id, centroid in group_centroids.items():
        indices = merged_clusters_info[cluster_id]['indices']
        data.loc[indices, 'distance_to_centroid'] = cosine_distances(doc_term_mat_xfm[indices], centroid.reshape(1, -1)).flatten()
        data.loc[indices, 'coherence_weight'] = normalized_coherences[cluster_id]
    
    features = np.hstack([doc_term_mat_xfm, data[['distance_to_centroid', 'coherence_weight']].fillna(0).values])
    labels = data['label'].map({'relevant': 1, 'irrelevant': 0, 'borderline': 2, 'unknown': -1})
    
    le = LabelEncoder()
    labels = le.fit_transform(data['label'])
    
    # Filter out unknown labels
    known_mask = data['label'] != 'unknown'
    features_known = features[known_mask]
    labels_known = labels[known_mask]
    
    return features, labels, features_known, labels_known, le

def train_and_evaluate_classifier(features, labels, le):
    if len(np.unique(labels)) == 1:
        print("Warning: All labels are the same. Cannot split or train model.")
        return DummyClassifier(strategy="constant", constant=labels[0])
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return clf

def predict_article_relevance(data, doc_term_mat_xfm, group_centroids, normalized_coherences, clf, le):
    """Predict relevance for all articles."""
    # Use the entire dataset
    data_copy = data.copy()

    # Initialize columns using .loc to avoid warnings
    data_copy.loc[:, 'nearest_centroid'] = None
    data_copy.loc[:, 'distance_to_nearest_centroid'] = np.inf

    for cluster_id, centroid in group_centroids.items():
        distances = cosine_distances(doc_term_mat_xfm, centroid.reshape(1, -1)).flatten()
        mask = distances < data_copy['distance_to_nearest_centroid']
        data_copy.loc[mask, 'nearest_centroid'] = cluster_id
        data_copy.loc[mask, 'distance_to_nearest_centroid'] = distances[mask]

    data_copy.loc[:, 'coherence_weight'] = data_copy['nearest_centroid'].map(normalized_coherences).fillna(0)

    features = np.hstack([
        doc_term_mat_xfm,
        data_copy[['distance_to_nearest_centroid', 'coherence_weight']].values
    ])

    predictions = clf.predict(features)
    probabilities = clf.predict_proba(features)
    
    data_copy.loc[:, 'predicted_label'] = le.inverse_transform(predictions)
    for i, class_name in enumerate(le.classes_):
        data_copy.loc[:, f'prob_{class_name}'] = probabilities[:, i]

    return data_copy

def categorize_articles(data, doc_term_mat_xfm, group_centroids, normalized_coherences):
    """Categorize all articles as Relevant, Irrelevant, or Borderline."""
    # Create an explicit copy
    categorized_data = data.copy()

    min_max_normalize = lambda x: (x - x.min()) / (x.max() - x.min())

    irrelevant_centroids = {cid: group_centroids[cid] for cid in irrelevant_clusters}

    categorized_data.loc[:, 'sim_to_nearest_irrelevant_centroid'] = -np.inf
    categorized_data.loc[:, 'nearest_irrelevant_centroid_coherence'] = 0

    for cluster_id, centroid in irrelevant_centroids.items():
        sims = cosine_similarity(doc_term_mat_xfm, centroid.reshape(1, -1)).flatten()
        mask = sims > categorized_data['sim_to_nearest_irrelevant_centroid']
        categorized_data.loc[mask, 'sim_to_nearest_irrelevant_centroid'] = sims[mask]
        categorized_data.loc[mask, 'nearest_irrelevant_centroid_coherence'] = normalized_coherences.get(cluster_id, 0)
    
    # Incorporate classifier probabilities
    categorized_data['classifier_score'] =  categorized_data['prob_relevant'] - categorized_data['prob_irrelevant']
    
    # Adjust final_score calculation
    categorized_data['final_score'] = (
        min_max_normalize(categorized_data['classifier_score']) * 0.5 +
        min_max_normalize(
            categorized_data['prob_relevant'] - 
            (categorized_data['sim_to_nearest_irrelevant_centroid'] * 
             categorized_data['nearest_irrelevant_centroid_coherence'])**(1/8)
        ) * 0.5
    )

    mean_score = categorized_data['final_score'].mean()
    std_score = categorized_data['final_score'].std()
    lower_threshold = mean_score - std_score
    upper_threshold = mean_score + std_score

    kmeans = KMeans(n_clusters=3, random_state=42).fit(categorized_data[['final_score']])
    centroids = kmeans.cluster_centers_.flatten()
    sorted_centroids = np.sort(centroids)

    final_lower_threshold = (lower_threshold + sorted_centroids[0]) / 2
    final_upper_threshold = (upper_threshold + sorted_centroids[-1]) / 2

    categorized_data['category'] = np.where(
        categorized_data['final_score'] <= final_lower_threshold, 'Irrelevant',
        np.where(categorized_data['final_score'] >= final_upper_threshold, 'Relevant', 'Borderline')
    )

    return categorized_data

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
    Export categorized data to a CSV file for user editing, with prepopulated user judgments.
    
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
    
    # Add a new column for user judgment and prepopulate based on category
    export_df['user_judgment'] = export_df['category'].map({
        'Irrelevant': 'D',
        'Relevant': 'K',
        'Borderline': 'K'
    })
    
    # Reorder columns to put user_judgment at the end
    columns_order = columns_to_export + ['user_judgment']
    export_df = export_df[columns_order]
    
    # Export to CSV
    export_df.to_csv(output_file, index=False)
    
    print(f"Data exported to {output_file}")
    print(f"User judgments prepopulated: 'D' for Irrelevant, 'K' for Relevant and Borderline")


def get_user_defined_clusters(config):
    """
    Read the user-defined cluster judgments from a CSV file and return lists of
    relevant, irrelevant, and borderline clusters.

    Args:
    config (dict): Configuration dictionary containing the file path.

    Returns:
    tuple: Lists of relevant, irrelevant, and borderline clusters.
    """
    # Get the file path from the config, with a default value
    file_path = config.get('cluster_judgments_file', 'data/text_review_clusters.csv')

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Initialize lists for each category
        relevant_clusters = []
        irrelevant_clusters = []
        borderline_clusters = []

        # Process each row
        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            judgment = row['user_judgment'].upper()

            # Convert numeric cluster IDs to integers
            if isinstance(cluster_id, (int, float)) or (isinstance(cluster_id, str) and cluster_id.isdigit()):
                cluster_id = int(cluster_id)
            
            # Categorize based on user judgment
            if judgment == 'R':
                relevant_clusters.append(cluster_id)
            elif judgment == 'I':
                irrelevant_clusters.append(cluster_id)
            elif judgment == 'B':
                borderline_clusters.append(cluster_id)

        return relevant_clusters, irrelevant_clusters, borderline_clusters

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return [], [], []
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        return [], [], []
    except KeyError as e:
        print(f"Error: The file is missing a required column: {e}")
        return [], [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], [], []

config = load_user_config()
data, latent_sa, doc_term_mat_xfm, terms, group_centroids, merged_clusters_info, sorted_all_coherences = load_data_and_clusters(config)

# Get user-defined clusters
relevant_clusters, irrelevant_clusters, borderline_clusters = get_user_defined_clusters(config)

data = label_articles(data, merged_clusters_info, relevant_clusters, irrelevant_clusters, borderline_clusters)

normalized_coherences = normalize_values(dict(sorted_all_coherences))

features, labels, features_known, labels_known, le = prepare_features(data, doc_term_mat_xfm, group_centroids, normalized_coherences)

clf = train_and_evaluate_classifier(features, labels, le)  # Train only on known labels

data_with_predictions = predict_article_relevance(data, doc_term_mat_xfm, group_centroids, normalized_coherences, clf, le)

categorized_data = categorize_articles(data_with_predictions, doc_term_mat_xfm, group_centroids, normalized_coherences)

export_categorized_data(categorized_data, output_file = normalize_path(config.get('categorized_output_file', './data/text_analysis/categorized_articles.csv')))

sample_articles(categorized_data)