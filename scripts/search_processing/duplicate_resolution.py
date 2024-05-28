#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 04: Duplicate Resolution

This script resolves duplicates in the preprocessed data based on manual review decisions.

1. Loads the preprocessed data and user configuration.
2. Processes any manual entries. These entries can be flagged as:
    - 'G': Grey Literature. These entries are marked as 'Grey Literature' in the 'grey_flag' column.
    - 'U': Update. These entries are updated during the manual review process.
    - 'R': Removal. These entries are removed during the manual review process.
3. Merges the preprocessed data and the removal log into a combined DataFrame.
4. Exports any entries flagged as potential duplicates to a CSV file for manual review.

NEXT STEP: You must review the flagged entries in the CSV file and make a final changes/decisions to the dataset.

Manual review flags (update_flag column):
- 'R': Removal. Entries with this flag are removed from the dataset.
- 'K': Keep. Entries with this flag are kept - sometimes breaking the 'duplicate' association.
- 'M': Merge. Entries with this flag are merged together to add additional information (keywords, abstracts, etc.).

flag_note:
- 'G': Grey Literature. Entries with this flag are marked as 'Grey Literature' in the 'grey_flag' column.
- 'H': Hollow Data. Entries with this flag are removed from the dataset.
- 'NOT DUP': Entries with this flag are not duplicates and are kept in the dataset.
- 'IRRELEVANT': Entries with this flag are irrelevant and are removed from the dataset.
- 'DUP': Entries with this flag are duplicates and are removed from the dataset.

Configuration parameters:
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
- quick_data_merge_threshold

Created on Tue Apr  2 17:27:27 2024

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_preproc import *


def create_combined_string(row):
    "Create a combined string from the row data"
    return ' '.join([str(row['year']), row['title'], row['abstract'], row['authors']])


def handle_removals(review_df, preproc_df, removal_log_df, action_log, config):
    """
    Handle removals of articles based on review decisions.
    
    Parameters:
    - review_df (pd.DataFrame): DataFrame containing review decisions.
    - preproc_df (pd.DataFrame): DataFrame containing preprocessed articles.
    - removal_log_df (pd.DataFrame): DataFrame logging removed articles.
    - action_log (dict): Log of actions taken.
    - config (dict): Configuration settings.
    
    Returns:
    - removal_log_df (pd.DataFrame): Updated removal log DataFrame.
    - action_log (dict): Updated log of actions taken.
    """
    
    removals = review_df[review_df['update_flag'].str.upper() == 'R']
    
    reason_map = {
        'H': 'Hollow data that cannot be recouped',
        'G': 'Grey literature',
        'Dup': 'Removed for being a duplicate',
        'Irrelevant': 'Not relevant to the search criteria'
    }
    
    indices_to_remove = []
    for _, row in removals.iterrows():
        reason = reason_map.get(row['flag_note'], 'Unknown reason')
        removed_article = preproc_df[preproc_df['orig_index'] == str(row['orig_index'])].assign(reason_for_removal=reason, removal_step='Duplicate Resolution')
        removal_log_df = pd.concat([removal_log_df, removed_article])
        indices_to_remove.append(str(row.name))
        
        action_log['removed'].append(str(row['orig_index']))
        
    # Remove duplicates from the preprocessed data
    preproc_df.drop(indices_to_remove, inplace=True)
    
    return removal_log_df, action_log


def handle_keeps(review_df, preproc_df, removal_log_df, action_log):
    """
    Handle keeps of articles based on review decisions.
    
    Parameters:
    - review_df (pd.DataFrame): DataFrame containing review decisions.
    - preproc_df (pd.DataFrame): DataFrame containing preprocessed articles.
    - removal_log_df (pd.DataFrame): DataFrame logging removed articles.
    - action_log (dict): Log of actions taken.
    
    Returns:
    - preproc_df (pd.DataFrame): Updated DataFrame with processed articles.
    - action_log (dict): Updated log of actions taken.
    """
    
    keeps = review_df[review_df['update_flag'].str.upper() == 'K']
    
    indices_to_remove = []
    if sum([idx in removal_log_df['orig_index'].values for idx in keeps.orig_index]):
        for _, row in keeps.iterrows():
            if row['orig_index'] in removal_log_df['orig_index'].values:
                kept_article = removal_log_df[removal_log_df['orig_index'] == str(row['orig_index'])]
                preproc_df = pd.concat([preproc_df, kept_article])
                removal_log_df = removal_log_df[removal_log_df['orig_index'] != str(row['orig_index'])]
                indices_to_remove.append(str(row.name))
                
                action_log['updated'].append(str(row['orig_index']))
                
    # Remove duplicates from the preprocessed data
    removal_log_df.drop(indices_to_remove, inplace=True)
    
    return preproc_df, action_log


def handle_merges(review_df, preproc_df, removal_log_df, config, action_log):
    """
    Handle merges of articles based on review decisions.
    
    Parameters:
    - review_df (pd.DataFrame): DataFrame containing review decisions.
    - preproc_df (pd.DataFrame): DataFrame containing preprocessed articles.
    - removal_log_df (pd.DataFrame): DataFrame logging removed articles.
    - config (dict): Configuration settings.
    - action_log (dict): Log of actions taken.
    
    Returns:
    - preproc_df (pd.DataFrame): Updated DataFrame with processed articles.
    - removal_log_df (pd.DataFrame): Updated removal log DataFrame.
    - action_log (dict): Updated log of actions taken.
    """
    
    merge_threshold = float(config.get('quick_data_merge_threshold', 0.95))
    groups = review_df[review_df['update_flag'].str.upper() == 'M'].groupby('group_id')
    
    for group_id, group_df in groups:
        if group_df['group_mean_sim'].iloc[0] > merge_threshold:
            preproc_df, removal_log_df = quick_merge(group_df, preproc_df, removal_log_df, config)
            action_log['merged'].append(group_id)
        else:
            preproc_df, removal_log_df = detailed_merge(review_df, group_df, preproc_df, removal_log_df, config)
            action_log['merged'].append(group_id)
    
    return preproc_df, removal_log_df, action_log


def quick_merge(group_df, preproc_df, removal_log_df, config, action_log):
    """
    Quickly merge articles in a group based on review decisions.
    
    Parameters:
    - group_df (pd.DataFrame): DataFrame containing articles to be merged.
    - preproc_df (pd.DataFrame): DataFrame containing preprocessed articles.
    - removal_log_df (pd.DataFrame): DataFrame logging removed articles.
    - config (dict): Configuration settings.
    - action_log (dict): Log of actions taken.
    
    Returns:
    - preproc_df (pd.DataFrame): Updated DataFrame with processed articles.
    - removal_log_df (pd.DataFrame): Updated removal log DataFrame.
    """
    
    is_preproc_list = [str(idx) in preproc_df['orig_index'] for idx in group_df['orig_index']]
    preproc_index = list(group_df[is_preproc_list]['orig_index'].values)
    
    if len(preproc_index) < 2:
        # Either no preproc indices or only one - either way, nothing to do here
        return preproc_df, removal_log_df
    
    # Assuming the first entry is the one to keep, use all the others
    indices_to_remove = []
    for orig_idx, row in group_df.loc[preproc_index[1:]].iterrows():
        article_to_remove = preproc_df[preproc_df['orig_index'].astype(str) == str(orig_idx)].copy()
        article_to_remove['reason_for_removal'] = 'Duplicate - removed due to quick merge'
        article_to_remove['removal_step'] = 'Duplicate Resolution'
        removal_log_df = pd.concat([removal_log_df, article_to_remove])
        indices_to_remove.append(str(orig_idx))

        # Update action log
        action_log['merged'].append(str(orig_idx))
    
    # Remove duplicates from the preprocessed data
    preproc_df.drop(indices_to_remove, inplace=True)

    return preproc_df, removal_log_df


def detailed_merge(review_df, group_df, preproc_df, removal_log_df, config, action_log):
    """
    Detailed merge articles in a group based on review decisions.
    
    Parameters:
    - review_df (pd.DataFrame): DataFrame containing review decisions.
    - group_df (pd.DataFrame): DataFrame containing articles to be merged.
    - preproc_df (pd.DataFrame): DataFrame containing preprocessed articles.
    - removal_log_df (pd.DataFrame): DataFrame logging removed articles.
    - config (dict): Configuration settings.
    - action_log (dict): Log of actions taken.
    
    Returns:
    - preproc_df (pd.DataFrame): Updated DataFrame with processed articles.
    - removal_log_df (pd.DataFrame): Updated removal log DataFrame.
    """
    
    # Initialize variables to store the merged information
    merged_journal = ''
    merged_keywords = set()
    merged_authors = set()
    longest_abstract = ''
    longest_abstract_len = 0
    primary_article = None
    set_of_years = set()
    set_of_dois = set()
    
    preproc_index = [str(idx) for idx in group_df['orig_index'] if str(idx) in preproc_df['orig_index']]

    # Iterate through each article in the group to consolidate information
    for orig_index, row in group_df.iterrows():
        article = preproc_df[preproc_df['orig_index'].astype(str) == str(orig_index)] \
            if str(orig_index) in preproc_index else \
            removal_log_df[removal_log_df['orig_index'].astype(str) == str(orig_index)]

        # Select the longest journal name
        if len(article['journal'].str.strip()) > len(merged_journal):
            merged_journal = article['journal'].str.strip().values[0]

        # Consolidate keywords, ensuring uniqueness
        article_keywords = set([kw.lower() for kw in article['keywords'].str.split(';').values[0] if kw])
        merged_keywords.update(article_keywords)
        
        # Consolidate authors, ensuring uniqueness
        article_authors = set([kw.lower() for kw in article['authors'].str.split(';').values[0] if kw])
        merged_authors.update(article_authors)
        
        # Update lists for checks
        set_of_years.add(article['year'].astype(int).astype(str).str.strip().values[0])
        set_of_dois.add(article['doi'].astype(str).str.strip().values[0])
        

        # Select the longest abstract
        if len(article['abstract']) > longest_abstract_len:
            longest_abstract = article['abstract'].str.strip().values[0]
            longest_abstract_len = len(article['abstract'])
            primary_article = article.copy()

    # Update the primary article with merged information
    new_idx = str(preproc_df.shape[0] + removal_log_df.shape[0])
    primary_article.index = [new_idx]
    
    primary_article.loc[new_idx, 'journal'] = merged_journal
    primary_article.loc[new_idx, 'keywords'] = '; '.join(sorted(merged_keywords))
    primary_article.loc[new_idx, 'authors'] = '; '.join(sorted(merged_authors))
    primary_article.loc[new_idx, 'abstract'] = longest_abstract
    primary_article['merged'] = True
    primary_article.loc[new_idx, 'orig_index'] = new_idx
    
    # Select and flag years
    set_of_years = set([year for year in set_of_years if year.isdigit() or (year.replace('.', '', 1).isdigit() and year.count('.') == 1)]) # remove blanks and non-digit entries
    num_years = len(set_of_years)
    if num_years != 1:
        review_df.loc[group_df.index, 'flag_years'] = True
        
    if num_years > 0:
        primary_article.loc[new_idx, 'year'] = min(set_of_years, key=int) # set earliest year
    
    # Select and flag dois
    set_of_dois = set([doi for doi in set_of_dois if doi != '']) # remove blanks
    num_dois = len(set_of_dois)
    if num_dois != 1:
        review_df.loc[group_df.index, 'flag_dois'] = True
        
    if num_dois > 0:
        primary_article.loc[new_idx, 'doi'] = max(set_of_dois, key=len) # set longest doi

    # Update the preprocessed DataFrame with the merged article
    preproc_df = pd.concat([preproc_df, primary_article])

    # Remove the other articles in the group from preproc_df and add them to removal_log_df
    filtered_df = preproc_df[preproc_df['orig_index'].isin(preproc_index)].copy()
    filtered_df['reason_for_removal'] = 'Duplicate - removed due to detailed merge'
    filtered_df['removal_step'] = 'Duplicate Resolution'
    removal_log_df = pd.concat([removal_log_df, filtered_df])
    
    # Update action log
    action_log['merged'].append(preproc_index)
    
    # Remove duplicates from the preprocessed data
    preproc_df.drop(preproc_index, inplace=True)

    return preproc_df, removal_log_df


def process_article_duplicates(preproc_df, removal_log_df, combined_df, config):
    """
    Process article duplicates based on review decisions, handling removals, keeps, and merges.
    
    Parameters:
    - preproc_df (pd.DataFrame): DataFrame containing preprocessed articles.
    - removal_log_df (pd.DataFrame): DataFrame logging removed articles.
    - config (dict): Configuration settings.
    
    Returns:
    - preproc_df (pd.DataFrame): Updated DataFrame with processed articles.
    - removal_log_df (pd.DataFrame): Updated removal log DataFrame.
    - action_log (dict): Log of actions taken on review_df items.
    """
    
    # Load the manual review DataFrame
    review_csv_path = config.get('duplicates_review_csv', './data/search_processing/duplicates_review.csv')
    review_df  = pd.read_csv(review_csv_path, encoding='latin1', index_col='orig_index')
    review_df  = replace_nulls(review_df, review_df.columns)
    review_df.index.name = 'index'
    review_df['orig_index'] = review_df.index
    
    action_log = {'updated': [], 'removed': [], 'merged': []}
    
    # Process removals ('R')
    removal_log_df, action_log = handle_removals(review_df, preproc_df, removal_log_df, action_log)
    
    # Process keeps ('K')
    preproc_df, action_log = handle_keeps(review_df, preproc_df, removal_log_df, action_log)
    
    # Process merges ('M')
    preproc_df, removal_log_df, action_log = handle_merges(review_df, preproc_df, removal_log_df, config, action_log)
    
    # Notify and save if reviews are needed
    if 'flag_years' in review_df.columns or 'flag_dois' in review_df.columns:
        to_review_bool = review_df['flag_years'] | review_df['flag_dois']
        if sum(to_review_bool) > 0:
            print(f"Files require review: {sum(to_review_bool)}. See the flag_years and flag_dois columns.")
            review_df.to_csv(review_csv_path, index=False)
    
    return preproc_df, removal_log_df, action_log



###
#   Start 
###

# Load preprocessed data
config = load_user_config()

preprocessed_df, removal_log_df = load_data(config)

# Combined info:
combined_df = pd.merge(preprocessed_df, removal_log_df, how='outer', on=list(set(preprocessed_df.columns) & set(removal_log_df.columns)), suffixes=('_preprocessed', '_removal'))
combined_df.fillna('', inplace=True)
combined_df.index = combined_df.orig_index
combined_df.index.name = 'index'
combined_df.sort_index(inplace = True)

# Resolve duplicates
preprocessed_df, removal_log_df, action_log = process_article_duplicates(preprocessed_df, removal_log_df, combined_df, config)

# Export action log to CSV
act_log_csv_path = config.get('actions_log_csv', './data/search_processing/actions_log.csv')
pd.DataFrame(action_log).to_csv(act_log_csv_path, index=False)

backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix='dupe_resolve')