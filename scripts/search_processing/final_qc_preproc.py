#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 05: FINAL QUALITY CHECKS

This script is the final quality control (QC) step in the data preprocessing pipeline:
1. Loads the user configuration and preprocessed data.
2. If a review CSV file exists, it processes entries marked for removal ('R') and addition ('A').
3. Removes 'hollow' data entries (entries with insufficient information) and grey literature.
4. Performs a series of quality checks on the data, flagging any entries that need review.
5. Backs up and saves the updated preprocessed data and removal log.

Quality checks include:
- Checking the length of the abstract.
- Checking the publication year.
- Checking the language of the article.
- Flagging articles with missing DOIs.

Configuration parameters:
- duplicates_review_csv
- hollow_flag
- min_abstract_length
- min_publication_year
- max_publication_year
- allowed_languages

NEXT STEP:
If any entries are flagged for review during the final QC, you should review these entries and the removal log.
Fix any issues directly in the preprocessed data and continue to the next step in the pipeline: text analysis.

Created on Tue Apr 2 23:05:09 2024

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_preproc import *


def check_abstract_length(df, min_length=100):
    """Check if the abstract length is within the specified range."""
    
    df['abstract_length'] = df['abstract'].apply(lambda x: len(x.split()))
    df['abstract_check']  = df['abstract_length'].apply(lambda x: x <= min_length)


def check_publication_year(df, min_year=1900, max_year=2024):
    """Check if the publication year is within the specified range."""
    df['year_check'] = ~df['year'].apply(lambda x: min_year <= int(x) <= max_year)


def check_language(df, config):
    """Check if the article's language is within the allowed list."""
    allowed_languages = config.get('languages', 'english').split(', ')
    df['language_check'] = df['language'].apply(lambda x: x.lower() not in allowed_languages if x else False)


def check_missing_doi(df):
    """Flag articles with missing DOIs."""
    df['doi_check'] = df['doi'].apply(lambda x: x.strip() == '')


def perform_quality_checks(df, config):
    """Perform all quality checks and flag entries for review.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing publication metadata.
    config : dict
        Configuration settings
    """
    
    check_abstract_length(df, min_length = int(config.get('min_length', 100)))
    check_publication_year(df, min_year = int(config.get('year_min', 1900)), max_year = int(config.get('year_max', 2024)))
    check_language(df, config)
    check_missing_doi(df)
    df['needs_review'] = (df['abstract_check'] | df['year_check'] | df['language_check'] | df['doi_check'])


def remove_flagged_entries(df, removal_log_df, flag_column, removal_reason, removal_step):
    """
    Remove entries flagged with a specific flag column and log the removal.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing publication metadata.
    removal_log_df : pd.DataFrame
        DataFrame containing the removal log.
    flag_column : str
        The name of the flag column to check for removal.
    removal_reason : str
        The reason for removal to be logged.
    removal_step : str
        The step in the process where removal is occurring.
    
    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with flagged entries removed.
    removal_log_df : pd.DataFrame
        Updated removal log DataFrame with logged removals.
    """
    # Ensure the flag column contains integers, handling non-numeric values
    df[flag_column] = pd.to_numeric(df[flag_column], errors='coerce').fillna(0).astype(int)
    
    # Identify indices of rows flagged for removal
    flagged_indices = df[df[flag_column] == 1].index
    
    # Create a DataFrame of rows to be removed, with reasons for removal
    to_append = df.loc[flagged_indices].assign(reason_for_removal=removal_reason, removal_step=removal_step)
    
    # Concatenate the removal log with the new removals
    removal_log_df = pd.concat([removal_log_df, to_append], ignore_index=True)
    
    # Drop the flagged rows from the original DataFrame
    df.drop(index=flagged_indices, inplace=True)
    
    return df, removal_log_df



###
#   Start 
###

# Load preprocessed data
config = load_user_config()

# Load preprocessed datas
preprocessed_df, removal_log_df = load_data(config)
review_csv_path = config.get('duplicates_review_csv', './data/search_processing/duplicates_review.csv')

# Load duplicates_review decisions if the file exists
if op.exists(review_csv_path):
    review_df = pd.read_csv(review_csv_path)
    
    # Process 'R' marked entries for removal
    removal_logs = []
    indices_to_remove = []
    for _, row in review_df[review_df['Decision'] == 'R'].iterrows():
        # Prepare the row to be appended to the removal log
        row_to_log = row.drop(['Decision']).to_dict()
        row_to_log.update({
            'reason_for_removal': 'Duplicate',
            'removal_step': 'Final Preprocessing'
        })
        removal_logs.append(row_to_log)
        
        # Collect indices to remove
        if row['orig_index'] in preprocessed_df['orig_index'].values:
            indices_to_remove.append(row['orig_index'])
    
    # Remove rows in one operation
    preprocessed_df = preprocessed_df[~preprocessed_df['orig_index'].isin(indices_to_remove)]
    
    if removal_logs:
        removal_log_df = pd.concat([removal_log_df, pd.DataFrame(removal_logs)], ignore_index=True)
    
    # Process 'A' marked entries for addition
    additions = []
    for _, row in review_df[review_df['Decision'] == 'A'].iterrows():
        # Ensure the orig_index is not already in preprocessed_df
        if row['orig_index'] not in preprocessed_df['orig_index'].values:
            # Drop 'Index' and 'Decision' columns and prepare the row to add
            row_to_add = row.drop(['Index', 'Decision']).to_frame().T
            additions.append(row_to_add)
    
    if additions:
        preprocessed_df = pd.concat([preprocessed_df] + additions, ignore_index=True)

# Remove hollow data
preprocessed_df, removal_log_df = remove_flagged_entries(
    preprocessed_df, removal_log_df, 
    flag_column='hollow_flag', 
    removal_reason='Hollow Data', 
    removal_step='Final Preprocessing'
)

# Remove grey data
preprocessed_df, removal_log_df = remove_flagged_entries(
    preprocessed_df, removal_log_df, 
    flag_column='grey_flag', 
    removal_reason='Grey Data', 
    removal_step='Final Preprocessing'
)

# Final quality check and flagging for review
perform_quality_checks(preprocessed_df, config)

# Update data files
backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix='fianl_qc')

num_flagged_entries = preprocessed_df.needs_review.sum()

if num_flagged_entries > 0:
    print(f"Final preprocessing complete. {num_flagged_entries} entries flagged for review. Please review flagged preprocessed entries and the removal log.")
else:
    print("Final preprocessing complete. Please review the removal log.")