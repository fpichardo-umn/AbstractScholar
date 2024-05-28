#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 05: FINAL QUALITY CHECKS

This script is the final quality control (QC) step in the data preprocessing pipeline:
1. Loads the user configuration and preprocessed data.
2. If a review CSV file exists, it processes entries marked for removal ('R') and addition ('A').
3. Removes 'hollow' data entries (entries with insufficient information).
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

NEXT STEP: If any entries are flagged for review during the final QC, you must review these entries and the removal log.

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
    df['abstract_check']  = df['abstract_length'].apply(lambda x: min_length <= x)


def check_publication_year(df, min_year=1900, max_year=2024):
    """Check if the publication year is within the specified range."""
    df['year_check'] = df['year'].apply(lambda x: min_year <= x <= max_year)


def check_language(df, config):
    """Check if the article's language is within the allowed list."""
    allowed_languages = config.get('languages', 'english').split(', ')
    df['language_check'] = df['language'].str.lower().isin(allowed_languages)


def check_missing_doi(df):
    """Flag articles with missing DOIs."""
    df['doi_check'] = df['doi'].apply(lambda x: x.strip() != '')


def perform_quality_checks(df, config):
    """Perform all quality checks and flag entries for review.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing publication metadata.
    config : dict
        Configuration settings
    """
    
    check_abstract_length(df, min_year = int(config.get('min_length', 100)), max_year = int(config.get('max_length', 50000)))
    check_publication_year(df, min_year = int(config.get('year_min', 1900)), max_year = int(config.get('year_max', 2024)))
    check_language(df, config)
    check_missing_doi(df)
    df['needs_review'] = ~(df['abstract_check'] & df['year_check'] & df['language_check'] & df['doi_check'])


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
    for index, row in review_df[review_df['Decision'] == 'R'].iterrows():
        removal_log_df = removal_log_df.append({'index': row['Index'], 'reason_for_removal': 'Duplicate', 'removal_step': 'Final Preprocessing'})
        preprocessed_df.drop(index=row['Index'], inplace=True)
    
    # Process 'A' marked entries for addition
    for index, row in review_df[review_df['Decision'] == 'A'].iterrows():
        preprocessed_df = preprocessed_df.append(row.drop(['Index', 'Decision']))

# Remove hollow data
hollow_indices = preprocessed_df[preprocessed_df['hollow_flag'] == True].index
removal_log_df = removal_log_df.append(preprocessed_df.loc[hollow_indices].assign(reason_for_removal='Hollow Data', removal_step='Final Preprocessing'))
preprocessed_df.drop(index=hollow_indices, inplace=True)

# Final quality check and flagging for review
preprocessed_df = perform_quality_checks(preprocessed_df, config)

# Update data files
backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix='fianl_qc')

if preprocessed_df.needs_review.sum() > 0:
    print("Final preprocessing complete. Please review flagged preprocessed entries and the removal log.")
else:
    print("Final preprocessing complete. Please review the removal log.")