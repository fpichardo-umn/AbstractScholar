#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 03: Duplicate Review

This script is responsible for identifying and flagging potential duplicate entries in the dataset.
1. Load the preprocessed data and user configuration.
2. Process any manual entries.
    - 'G': Grey Literature. Entries with this flag are marked as 'Grey Literature' in the 'grey_flag' column.
    - 'U': Update. Entries with this flag are updated during the manual review process.
    - 'R': Removal. Entries with this flag are removed during the manual review process.
3. Merge the preprocessed data and the removal log into a combined DataFrame.
4. Hash the articles in the combined DataFrame.
5. Group similar articles together based on their hashes.
6. Export any entries flagged as potential duplicates to a CSV file for manual review.

NEXT STEP: User must review the flagged entries in the CSV file and update the relevant columns.
Manual review flags (update_flag column):
- 'R': Removal. Entries with this flag are removed from the dataset.
- 'K': Keep. Entries with this flag are kept - sometimes breaking the 'duplicate' association.
- 'M': Merge. Entries with this flag are merged together to add aditional information (keywords, abstracts, etc.).
flag_note:
- 'G': Grey Literature. Entries with this flag are marked as 'Grey Literature' in the 'grey_flag' column.
- 'H': Hollow Data. Entries with this flag are removed from the dataset.
- 'NOT DUP': Entries with this flag are not duplicates and are kept in the dataset.
- 'IRRELEVANT': Entries with this flag are irrelevant and are removed from the dataset.
- 'DUP': Entries with this flag are duplicates and are removed from the dataset.

Configuration entries:
   - search_results_path
   - preprocessed_file_path
   - removal_log_path
   - overwrite
   - hollow_threshold
   - hollow_check_cols
   - standard_text_removal_list
   - standard_text_removal_cols
   - doi_duplicate_removal_threshold
   - doi_recovery_threshold
   - doi_retrieval_cols
   - doi_retrieval_threshold
   - backup_path

Created on Tue Apr  2 17:27:27 2024

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

from itertools import combinations
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_search import *


def create_combined_string(row):
    return ' '.join([str(row['year']), row['title'], row['abstract'], row['authors']])


def hash_articles(df, config):
    """
    Hashes articles using Locality-Sensitive Hashing.

    Parameters:
    - df (pd.DataFrame): DataFrame containing articles.
    - config (dict): Configuration settings.

    Returns:
    - MinHashLSH: LSH object with articles hashed.
    """
    
    try:
        lsh_thresh = float(config.get('lsh_thresh', 0.5))
    except ValueError:
        raise ValueError(f"lsh_thresh value in config file is not a float: {config.get('lsh_thresh')}")
    
    try:
        lsh_num_perm = int(config.get('lsh_num_perm', 200))
    except ValueError:
        raise ValueError(f"lsh_thresh value in config file is not a float: {config.get('lsh_thresh')}")
    
    lsh = MinHashLSH(threshold=lsh_thresh, num_perm=lsh_num_perm)
    
    for idx in df.orig_index:
        combined_str = create_combined_string(df.loc[idx])
        m = MinHash(num_perm=lsh_num_perm)
        for d in combined_str.split():
            m.update(d.encode('utf8'))
        lsh.insert(f"doc_{idx}", m)

    return lsh


def identify_cohesive_subsets(similarity_matrix, config):
    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix
    
    # Apply HDBSCAN
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.55, metric='precomputed', linkage='complete')
    clusterer.fit(distance_matrix)
    
    # Extract the labels indicating cluster assignments
    labels = clusterer.labels_
    
    # Identify indices for documents in each cluster
    subsets = [np.where(labels == i)[0].tolist() for i in set(labels) if i != -1]
    
    return subsets


def append_similarity_results(group_data, similarity_matrix, mean_sim, results, map_art_to_group_idx, group_id, hollow_flag = False):
    """
    Append similarity results for a group or subgroup to the results list.
    Each row includes the group mean similarity.
    """
    group_indices = group_data.index.tolist()
    map_art_to_group_idx.update({art_idx: group_id for _, art_idx in enumerate(group_data.index.values)})

    for idx, row_idx in enumerate(group_indices):
        for jdx, col_idx in enumerate(group_indices):
            if idx < jdx:  # Ensure unique pairwise comparisons
                similarity_score = similarity_matrix[idx, jdx]
                results.append({
                    'group_id': group_id,
                    'article_1': f'doc_{row_idx}',
                    'article_2': f'doc_{col_idx}',
                    'similarity_score': similarity_score,
                    'group_mean_sim': mean_sim,
                    'hollow_group_flag': hollow_flag
                })


def process_subset_as_group(subset, group_data, tfidf_vectorizer, results, map_art_to_group_idx, group_id):
    """
    Process each identified subset as a separate group, calculating and appending similarity scores to results.
    """
    subset_group_data = group_data.iloc[subset]
    combined_strings = subset_group_data.apply(lambda row: create_combined_string(row), axis=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_strings)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    mean_sim = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    append_similarity_results(subset_group_data, similarity_matrix, mean_sim, results, map_art_to_group_idx, group_id)



def calculate_group_similarity(df, groups, config):
    """
    Calculate similarity scores between articles within groups and flag hollow groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing articles.
    groups : list of lists
        Groups of article indices identified as similar.
    config : dict
        Configuration settings.

    Returns
    -------
    pd.DataFrame
        DataFrame with similarity scores for article comparisons within groups, and flags for hollow groups.
    dict
        Mapping of article indices to their corresponding group IDs.
    """

    results = []  # List to collect data before creating DataFrame
    map_art_to_group_idx = {}  # Dictionary to map article indices to group IDs
    
    group_id = 0
    for group in groups:
        if len(group) > 1:
            # Extracting the data for the current group and creating a combined string
            group_data = df.loc[[doc.replace('doc_', '') for doc in group]]
            combined_strings = group_data.apply(lambda row: create_combined_string(row), axis=1)
            
            # Check if the group is hollow
            is_hollow = combined_strings.str.len().mean() < float(config.get('length_threshold', 20))  # Using config dict for threshold
            
            if is_hollow:
                # For hollow groups, all similarity scores and group mean similarity are set to 0, and flagged as hollow
                append_similarity_results(group_data, np.zeros((len(group),len(group))), 0, results, map_art_to_group_idx, group_id, hollow_flag = True)
            else:
                # Calculate TF-IDF and cosine similarity for non-hollow groups
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(combined_strings)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Determine if any of these groups should be split or removed
                mean_sim = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                if mean_sim < float(config.get('split_threshold', 0.90)):
                    subsets = identify_cohesive_subsets(similarity_matrix, config)
                   
                    if len(subsets) == len(group):
                        continue # This is not a cohesive group at all and should not be included
                    elif len(subsets) > 1:
                        # Split into subgroups
                        for subset in subsets:
                            # If the subset is only a single item, we can skip adding it, as it's not a group
                            if type(subset) == list and len(subset) > 1:
                                process_subset_as_group(subset, group_data, tfidf_vectorizer, results, map_art_to_group_idx, group_id)
                                group_id += 1  # Ensure each subset gets a unique group ID
                        continue
                    else: 
                        # there is only one group and usual processing
                        append_similarity_results(group_data, similarity_matrix, mean_sim, results, map_art_to_group_idx, group_id)
                else:
                        append_similarity_results(group_data, similarity_matrix, mean_sim, results, map_art_to_group_idx, group_id)
            group_id += 1
    
    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results, columns=['group_id', 'article_1', 'article_2', 'similarity_score', 'hollow_group_flag', 'group_mean_sim'])
    
    return results_df, map_art_to_group_idx


def merge_overlapping_groups(groups):
    """
    Merges overlapping groups to ensure that each document is assigned to a single group.

    Parameters:
    - groups (list of lists): Initial list of document groups with potential overlaps.

    Returns:
    - list of lists: Merged list of document groups without overlaps.
    """
    merged_groups = []
    for group in groups:
        # Find an existing group that overlaps with the current group
        overlapping_group = next((merged_group for merged_group in merged_groups if set(merged_group) & set(group)), None)
        if overlapping_group:
            # Merge the current group with the overlapping group
            overlapping_group.extend(doc for doc in group if doc not in overlapping_group)
        else:
            # Add the current group as a new group if no overlap was found
            merged_groups.append(group)
    return merged_groups


def group_similar_articles(lsh, df, config):
    """
    Groups similar articles and calculates similarity scores.

    Parameters:
    - lsh (MinHashLSH): LSH object with hashed articles.
    - df (pd.DataFrame): DataFrame containing articles.
    - config (dict): Configuration settings.

    Returns:
    - pd.DataFrame: DataFrame with groups and similarity scores.
    """
    
    
    try:
        lsh_num_perm = int(config.get('lsh_num_perm', 200))
    except ValueError:
        raise ValueError(f"lsh_thresh value in config file is not a float: {config.get('lsh_thresh')}")
    
    groups = []
    for idx in df.orig_index:
        combined_str = create_combined_string(df.loc[idx])
        m = MinHash(num_perm=lsh_num_perm)
        for d in combined_str.split():
            m.update(d.encode('utf8'))
        result = lsh.query(m)
        if result:
            groups.append(result)
    
    # Merge overlapping groups to ensure each document is only in one group
    groups = merge_overlapping_groups(groups)
            
    # Calculate within group similarities and mapping from art to group
    return calculate_group_similarity(df, groups, config)


def export_for_review(group_df, map_art_to_group_idx, combined_df, config):
    
    # Step 1: Normalize article identifiers to match combined_df's index and ensure decimal range for similarity scores
    group_df['article_1_idx'] = group_df['article_1'].apply(lambda x: str(int(x.replace('doc_', ''))))
    group_df['article_2_idx'] = group_df['article_2'].apply(lambda x: str(int(x.replace('doc_', ''))))
    group_df['similarity_score'] = group_df['similarity_score'].round(2)  # Ensure decimal range for similarity scores
    group_df['group_mean_sim'] = group_df['group_mean_sim'].round(2)  # Ensure decimal range for similarity scores
    
    # Step 2: Merge to enrich with article metadata and prepare data for similarity string
    enriched_df = pd.concat([
        group_df.rename(columns={'article_1_idx': 'article_idx', 'article_2_idx': 'comparison_idx', 'similarity_score': 'similarity_str'}),
        group_df.rename(columns={'article_2_idx': 'article_idx', 'article_1_idx': 'comparison_idx', 'similarity_score': 'similarity_str'})
    ])
    enriched_df['similarity_str'] = enriched_df.apply(lambda row: f"doc_{row['comparison_idx']}: {row['similarity_str']}", axis=1)
    
    # Step 3: Aggregate and deduplicate similarity strings within groups for each article
    sim_str_df = enriched_df.groupby('article_idx')['similarity_str'].apply(lambda x: '; '.join(sorted(set(x), key=lambda s: float(s.split(': ')[1]), reverse=True)))
    
    # Step 4: Merge similarity strings back to combined_df
    combined_df['article_idx'] = combined_df.index
    combined_df = combined_df.merge(sim_str_df, left_on='article_idx', right_index=True, how='left').rename(columns={'similarity_str': 'art_to_art_group_sim'})
    
    hollow_flag_map = group_df.drop_duplicates('group_id').set_index('group_id')['hollow_group_flag']
    combined_df['hollow_group_flag'] = combined_df['article_idx'].astype(str).map(map_art_to_group_idx).map(hollow_flag_map).fillna(False)
    
    # Step 5: Prepare the final DataFrame for review
    review_df = combined_df[['title', 'year', 'journal', 'authors', 'abstract', 'doi', 'keywords', 'orig_abstract', 'orig_index', 'hollow_flag', 'grey_flag', 'article_idx', 'reason_for_removal', 'removal_step', 'dup_doi_cnt', 'art_to_art_group_sim', 'hollow_group_flag']].copy()
    review_df['update_flag'] = ''
    review_df['flag_note'] = ''
    review_df['group_flag'] = ''
    review_df['group_id'] = review_df['article_idx'].map(map_art_to_group_idx).fillna(-1)  # Use -1 for articles without a group
    review_df['group_mean_sim'] = review_df['group_id'].map(group_df.groupby('group_id')['similarity_score'].mean())  # Compute mean similarity per group
    
    # Drop rows with group ID equal to -1
    review_df = review_df[review_df['group_id'] != -1]    
    
    # Convert group ID to integer type
    #review_df['group_id'] = review_df['group_id'].astype(int)
    
    # Get the count of articles within each group
    group_counts = review_df['group_id'].value_counts()
    
    # Map the group counts to the corresponding group IDs
    review_df['group_count'] = review_df['group_id'].map(group_counts)
    review_df = review_df.sort_values(by=['group_mean_sim', 'group_id', 'doi', 'dup_doi_cnt', 'removal_step', 'art_to_art_group_sim'])
    
    # Export to CSV
    review_csv_path = config.get('duplicates_review_csv', './data/search_processing/duplicates_review.csv')
    review_df.to_csv(review_csv_path, index=False)

    print(f"Potential duplicates exported for manual review to '{review_csv_path}'.")



###
#   Start 
###

# Load preprocessed data
config = load_user_config()

preprocessed_df, removal_log_df = load_data(config)

# Process data retrieval review flags
preprocessed_df, removal_log_df = process_manual_entries(preprocessed_df, removal_log_df, config, config.get('manual_retrieval_csv', './data/search_processing/manual_data_retrieval.csv'))

# Combined info:
# Merge the two DataFrames without changing the originals
combined_df = pd.merge(preprocessed_df, removal_log_df, how='outer', on=list(set(preprocessed_df.columns) & set(removal_log_df.columns)), suffixes=('_preprocessed', '_removal'))
combined_df.fillna('', inplace=True)
combined_df.index = combined_df.orig_index
combined_df.index.name = 'index'
combined_df.sort_index(inplace = True)

# Hash articles using LSH
lsh = hash_articles(combined_df, config)

# Group similar articles and calculate similarity scores
dup_groups_df, map_art_to_group_idx = group_similar_articles(lsh, combined_df, config)

# Export flagged entries for manual review
export_for_review(dup_groups_df, map_art_to_group_idx, combined_df, config)