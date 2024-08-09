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
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_text import *


def min_max_normalize(x):
    if isinstance(x, pd.Series):
        return (x - x.min()) / (x.max() - x.min())
    else:
        return MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()

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

def label_articles(data, relevant_clusters, irrelevant_clusters, borderline_clusters):
    data['label'] = 'unknown'  # Default label
    for cluster in relevant_clusters:
        data.loc[data.index[data.cluster == cluster], 'label'] = 'relevant'
    for cluster in irrelevant_clusters:
        data.loc[data.index[data.cluster == cluster], 'label'] = 'irrelevant'
    for cluster in borderline_clusters:
        data.loc[data.index[data.cluster == cluster], 'label'] = 'borderline'
    return data

def prepare_features(data, doc_term_mat_xfm, group_centroids, normalized_coherences, relevant_clusters):
    """Prepare features for the classifier."""
    for cluster_id, centroid in group_centroids.items():
        indices = merged_clusters_info[cluster_id]['indices']
        data.loc[indices, 'distance_to_centroid'] = cosine_distances(doc_term_mat_xfm[indices], centroid.reshape(1, -1)).flatten()
        data.loc[indices, 'coherence_weight'] = normalized_coherences[cluster_id]
    
    # Add similarity to relevant clusters feature
    relevant_indices = data[data['cluster'].astype(str).isin(relevant_clusters)].index
    
    relevant_vectors = doc_term_mat_xfm[relevant_indices]
    similarities = cosine_similarity(doc_term_mat_xfm, relevant_vectors)
    max_similarities = similarities.max(axis=1)
    
    features = np.hstack([
        doc_term_mat_xfm, 
        data[['distance_to_centroid', 'coherence_weight']].fillna(0).values,
        max_similarities.reshape(-1, 1)
    ])
    
    labels = data['label'].map({'relevant': 1, 'irrelevant': 0, 'borderline': 2, 'unknown': -1})
    
    le = LabelEncoder()
    labels = le.fit_transform(data['label'])
    
    # Filter out unknown labels
    known_mask = data['label'] != 'unknown'
    features_known = features[known_mask]
    labels_known = labels[known_mask]
    
    return features, labels, features_known, labels_known, le


def train_multiple_classifiers(features, labels, random_state_clf, n_splits=5, n_estimators=10, max_samples=0.8, max_features=0.8):
    classifiers = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state_clf)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=random_state_clf)),
        ('svm', SVC(probability=True, random_state=random_state_clf)),
        ('nb', GaussianNB()),
        ('lr', LogisticRegression(random_state=random_state_clf)),
        ('dt', DecisionTreeClassifier(random_state=random_state_clf)),
        ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state_clf)))
    ]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state_clf)
    
    all_predictions = []
    f1_scores = []
    for name, classifier in classifiers:
        predictions = np.zeros((len(features), len(np.unique(labels))))
        all_y_test = []
        all_y_pred = []
        
        for train_index, test_index in skf.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            if name == 'lr':
                # LogisticRegression doesn't need bagging
                classifier.fit(X_train, y_train)
                fold_pred = classifier.predict_proba(X_test)
                y_pred = classifier.predict(X_test)
            else:
                bagging_clf = BaggingClassifier(
                    estimator=classifier,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    max_features=max_features,
                    random_state=random_state_clf
                )
                bagging_clf.fit(X_train, y_train)
                fold_pred = bagging_clf.predict_proba(X_test)
                y_pred = bagging_clf.predict(X_test)
            
            predictions[test_index] = fold_pred
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
        
        # Calculate and print performance metrics
        accuracy = accuracy_score(all_y_test, all_y_pred)
        precision = precision_score(all_y_test, all_y_pred, average='weighted')
        recall = recall_score(all_y_test, all_y_pred, average='weighted')
        f1 = f1_score(all_y_test, all_y_pred, average='weighted')
        f1_scores.append(f1)

        print(f"Classifier: {name}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("--------------------")
        
        all_predictions.append(predictions)
    
    return all_predictions, f1_scores


def combine_predictions(all_predictions, features, group_centroids, normalized_coherences, doc_term_mat_xfm, irrelevant_clusters, f1_scores):
    # Calculate weights based on F1 scores
    weights = calculate_weights(f1_scores)
    
    # Weighted average of predictions
    weighted_predictions = np.average(all_predictions, axis=0, weights=weights)
    
    # Create a DataFrame with the weighted predictions
    prediction_df = pd.DataFrame(index=range(doc_term_mat_xfm.shape[0]))
    
    # Initialize columns with zeros
    prediction_df['prob_irrelevant'] = 0
    prediction_df['prob_relevant'] = 0
    prediction_df['prob_borderline'] = 0
    
    # Fill in predictions for known samples
    known_indices = np.where(np.any(weighted_predictions != 0, axis=1))[0]
    prediction_df.loc[known_indices, 'prob_irrelevant'] = weighted_predictions[known_indices, 0]
    prediction_df.loc[known_indices, 'prob_relevant'] = weighted_predictions[known_indices, 1]
    prediction_df.loc[known_indices, 'prob_borderline'] = weighted_predictions[known_indices, 2]
    
    # The rest of the function remains the same...
    prediction_df['classifier_score'] = prediction_df['prob_relevant'] - prediction_df['prob_irrelevant']
    
    # Add distance to nearest irrelevant centroid and its coherence
    irrelevant_centroids = {cid: group_centroids[cid] for cid in group_centroids.keys() if str(cid) in irrelevant_clusters}
    prediction_df['sim_to_nearest_irrelevant_centroid'] = -np.inf
    prediction_df['nearest_irrelevant_centroid_coherence'] = 0
    
    for cluster_id, centroid in irrelevant_centroids.items():
        sims = cosine_similarity(doc_term_mat_xfm, centroid.reshape(1, -1)).flatten()
        mask = sims > prediction_df['sim_to_nearest_irrelevant_centroid']
        prediction_df.loc[mask, 'sim_to_nearest_irrelevant_centroid'] = sims[mask]
        prediction_df.loc[mask, 'nearest_irrelevant_centroid_coherence'] = normalized_coherences.get(cluster_id, 0)
    
    # Calculate final score
    prediction_df['final_score'] = (
        min_max_normalize(prediction_df['classifier_score']) * 0.5 +
        min_max_normalize(
            prediction_df['prob_relevant'] - 
            (prediction_df['sim_to_nearest_irrelevant_centroid'] * 
             prediction_df['nearest_irrelevant_centroid_coherence'])**(1/8)
        ) * 0.5
    )
    
    # Handle any potential NaN values in the final_score
    prediction_df['final_score'] = prediction_df['final_score'].fillna(0)
    
    return prediction_df

def calculate_weights(f1_scores):
    total = sum(f1_scores)
    return [score / total for score in f1_scores]


def categorize_articles(prediction_df, random_state_clf):
    mean_score = prediction_df['final_score'].mean()
    std_score = prediction_df['final_score'].std()
    lower_threshold = mean_score - std_score
    upper_threshold = mean_score + std_score

    kmeans = KMeans(n_clusters=3, random_state=random_state_clf, n_init = 10).fit(prediction_df[['final_score']])
    centroids = kmeans.cluster_centers_.flatten()
    sorted_centroids = np.sort(centroids)

    final_lower_threshold = (lower_threshold + sorted_centroids[0]) / 2
    final_upper_threshold = (upper_threshold + sorted_centroids[-1]) / 2

    prediction_df['category'] = np.where(
        prediction_df['final_score'] <= final_lower_threshold, 'Irrelevant',
        np.where(prediction_df['final_score'] >= final_upper_threshold, 'Relevant', 'Borderline')
    )

    return prediction_df

def sample_articles(unlabeled_data):
    """Sample articles from each category."""
    relevant_sample = unlabeled_data[unlabeled_data['category'] == 'Relevant'][['title', 'abstract']].sample(5)
    borderline_sample = unlabeled_data[unlabeled_data['category'] == 'Borderline'][['title', 'abstract']].sample(5)
    irrelevant_sample = unlabeled_data[unlabeled_data['category'] == 'Irrelevant'][['title', 'abstract']].sample(5)

    print("Relevant Articles Sample:\n", relevant_sample)
    print("\nBorderline Articles Sample:\n", borderline_sample)
    print("\nIrrelevant Articles Sample:\n", irrelevant_sample)


def export_categorized_data(categorized_data, config):
    """
    Export categorized data to a CSV file for user editing, with improved prepopulated user judgments.
    
    Args:
    categorized_data (pd.DataFrame): The DataFrame containing categorized article data.
    config (dict): Configuration dictionary containing user supplied parameters.
    
    Returns:
    None
    """
    output_file = normalize_path(config.get('categorized_data_file', './data/text_analysis/categorized_articles.csv'))
    use_search_query = config.get('use_search_query', False)
    
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
    
    # Calculate the median final_score for borderline cases
    borderline_median = export_df[export_df['category'] == 'Borderline']['final_score'].median()
    
    # Process search query if enabled in config
    if use_search_query:
        search_query_file = normalize_path(config.get('search_query_file', './search_query.txt'))
        processed_terms = process_search_terms(search_query_file)
    
    # Add a new column for user judgment and prepopulate based on category and final_score
    def assign_judgment(row):
        if row['category'] == 'Irrelevant':
            if use_search_query:
                article_text = f"{row['title']} {row['journal']} {row['abstract']} {row['keywords']}"
                if article_matches_query(article_text, processed_terms):
                    return 'K'  # Change to 'Keep' if it matches any search term
            return 'D'
        elif row['category'] == 'Relevant':
            return 'K'
        else:  # Borderline
            return 'D' if row['final_score'] < borderline_median else 'K'
    
    export_df['user_judgment'] = export_df.apply(assign_judgment, axis=1)
    
    # Reorder columns to put user_judgment at the end
    columns_to_export.remove('title')
    columns_order = columns_to_export + ['user_judgment', 'title']
    export_df = export_df[columns_order]
    
    # Export to CSV
    export_df.to_csv(output_file, index=False)
    
    print(f"Data exported to {output_file}")
    print(f"User judgments prepopulated:")
    print("- 'D' for Irrelevant")
    print("- 'K' for Relevant")
    print("- 'D' for Borderline cases with final_score below the median")
    print("- 'K' for Borderline cases with final_score above or equal to the median")
    if use_search_query:
        print("- 'K' for Irrelevant cases that match the search query")


def get_user_defined_clusters(config):
    """
    Read the user-defined cluster judgments from a CSV file and return lists of
    relevant, irrelevant, and borderline clusters as individual cluster names.

    Args:
    config (dict): Configuration dictionary containing the file path.

    Returns:
    tuple: Lists of relevant, irrelevant, and borderline clusters as individual cluster names.
    """
    file_path = config.get('cluster_judgments_file', 'data/text_review_clusters.csv')

    def process_cluster_id(cluster_id):
        if isinstance(cluster_id, (int, float)) or (isinstance(cluster_id, str) and not cluster_id.startswith('merged')):
            return [str(int(cluster_id))]
        elif isinstance(cluster_id, str) and cluster_id.startswith('merged'):
            return [num for num in cluster_id.split('_')[1:]]
        else:
            return []

    try:
        df = pd.read_csv(file_path)

        relevant_clusters = []
        irrelevant_clusters = []
        borderline_clusters = []

        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            judgment = row['user_judgment'].upper()
            
            cluster_names = process_cluster_id(cluster_id)
            
            if judgment == 'R':
                relevant_clusters.extend(cluster_names)
            elif judgment == 'I':
                irrelevant_clusters.extend(cluster_names)
            elif judgment == 'B':
                borderline_clusters.extend(cluster_names)

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
    
def combine_data_and_predictions(data, categorized_data):
    """
    Combine the original data DataFrame with the categorized_data DataFrame.
    
    Args:
    data (pd.DataFrame): The original data DataFrame.
    categorized_data (pd.DataFrame): The DataFrame with predictions and categories.
    
    Returns:
    pd.DataFrame: Combined DataFrame with original data and new predictions/categories.
    """
    # Reset index of both DataFrames to ensure proper alignment
    data = data.reset_index(drop=True)
    categorized_data = categorized_data.reset_index(drop=True)
    
    # Combine the DataFrames
    combined_df = pd.concat([data, categorized_data], axis=1)
    
    # Remove duplicate columns if any
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    return combined_df


def process_search_terms(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split into terms, ignoring "AND"
    terms = re.split(r'\s+OR\s+|\s+AND\s+|\s+', content)
    
    processed_terms = []
    for term in terms:
        term = term.strip().strip('"')
        if not term:
            continue
        if '*' in term:
            # Handle wildcards
            term = term.replace('*', '.*')
        else:
            # Allow partial matches for non-wildcard terms
            term = f'.*{re.escape(term)}.*'
        processed_terms.append(term)
    
    return processed_terms


def article_matches_query(article_text, processed_terms):
    return any(re.search(term, article_text, re.IGNORECASE) for term in processed_terms)



config = load_user_config()
data, latent_sa, doc_term_mat_xfm, terms, group_centroids, merged_clusters_info, sorted_all_coherences = load_data_and_clusters(config)

# Set random state for reproducibility
random_state_clf = 42

# Get user-defined clusters
relevant_clusters, irrelevant_clusters, borderline_clusters = get_user_defined_clusters(config)

data = label_articles(data, relevant_clusters, irrelevant_clusters, borderline_clusters)

normalized_coherences = normalize_values(dict(sorted_all_coherences))

features, labels, features_known, labels_known, le = prepare_features(data, doc_term_mat_xfm, group_centroids, normalized_coherences, relevant_clusters)

all_predictions, f1_scores = train_multiple_classifiers(features_known, labels_known, random_state_clf)

prediction_df = combine_predictions(all_predictions, features, group_centroids, normalized_coherences, doc_term_mat_xfm, irrelevant_clusters, f1_scores)

categorized_data = categorize_articles(prediction_df, random_state_clf)

combined_df = combine_data_and_predictions(data, categorized_data)

export_categorized_data(combined_df, config)

#sample_articles(categorized_data)