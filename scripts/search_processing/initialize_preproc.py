#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 01: Initialize Preprocessing

This script, initialize_preproc.py, is designed to initialize the preprocessing of a collection of scientific articles. The script works with a DataFrame of articles, where each row represents a unique article and columns include details such as 'journal', 'keywords', 'authors', 'abstract', 'year', and 'doi'. 

The script performs the following main tasks:
1. Load User Configuration: The script begins by loading the user configuration which contains various parameters that control the preprocessing steps.
2. Initialize Files: The script initializes the files that will be used throughout the procedure. This includes the preprocessed data and the removal log.
3. Remove Standard Text: The script removes standard text from the articles based on the configuration parameters.
4. Flag Hollow Entries: The script flags hollow entries in the DataFrame. These are entries that lack essential information and are flagged for further review.
5. DOI Duplicate Flagging/Removal: The script performs an initial round of DOI duplicate flagging and removal. It consolidates the information of duplicate articles into a single entry.
6. Auto-Recover DOIs: The script attempts to auto-recover DOIs for entries that lack them. It uses a predefined strategy based on the configuration parameters.
7. Flag Missing DOIs: The script flags for the user the DOIs that they have to look up themselves. These are entries where the auto-recovery of DOIs was not successful.

NEXT STEP: User must attempt to retrieve missing DOIs manually and update the 'doi' column in the preprocessed data file. The user should then proceed to the next step in the preprocessing pipeline (data_rerieval.py).
Manual review flags:
- 'G': Grey Literature. Entries with this flag are marked as 'Grey Literature' in the 'grey_flag' column.
- 'U': Update. Entries with this flag are updated during the manual review process.
- 'R': Removal. Entries with this flag are removed during the manual review process.

The script uses several helper functions and classes from the 'scripts' directory, including functions from the 'gen_utils_preproc.py' module.

This script is part of a larger project aimed at analyzing a collection of scientific articles to extract meaningful insights and patterns.

The following configuration parameters are expected to be provided in the user configuration file:
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

Created on Mon Apr  1 21:37:53 2024

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_preproc import *


def initialize_preprocessing(config):
    """
    Initializes preprocessing by setting up necessary files.
    
    Parameters:
    -----------
    config : dict
        User configuration information
    
    Returns:
    --------
    pd.DataFrame
        preprocessed DataFrame
    pd.DataFrame
        removal log DataFrame
    """
    
    # Paths from config
    search_results_path = config.get('search_results_path', './data/search_results.csv')
    preprocessed_path   = config.get('preprocessed_file_path', './data/preprocessed_data.csv')
    removal_log_path    = config.get('removal_log_path', './data/search_processing/removal_log.csv')
    
    # Load search results and create preprocessed data file
    if (not os.path.exists(preprocessed_path)) or config.get('overwrite', False):
        preprocessed_df = pd.read_csv(search_results_path)
        
        # make sure the columns are correct in the search results
        preprocessed_df = check_cols_and_rename(preprocessed_df)
        preprocessed_df = replace_nulls(preprocessed_df, ['title', 'abstract', 'authors', 'journal', 'language', 'keywords', 'issue', 'pages', 'doi'])

        preprocessed_df['orig_abstract'] = preprocessed_df['abstract'].copy()
        preprocessed_df.abstract = preprocessed_df.abstract.str.lower()
        
        preprocessed_df['orig_index'] = preprocessed_df.index
        
        preprocessed_df.to_csv(preprocessed_path, index=False)
        # Convert all column names to lower case
        preprocessed_df.columns = preprocessed_df.columns.str.lower()
    else:
        raise FileExistsError(f"Overwrite is set to false and {preprocessed_path} exists already.")
    
    # Initialize removal log
    if (not os.path.exists(removal_log_path)) or config.get('overwrite', False):
        columns = list(preprocessed_df.columns) + ['reason_for_removal', 'removal_step']
        removal_log_df = pd.DataFrame(columns=columns)
        removal_log_df.to_csv(removal_log_path, index=False)
    else:
        raise FileExistsError(f"Overwrite is set to false and {removal_log_path} exists already.")
    
    return preprocessed_df, removal_log_df


def check_cols_and_rename(df):
    """
    Check columns and rename them to standardized names.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing article metadata.

    Returns:
    --------
    DataFrame
        DataFrame with standardized column names.
    """
    
    column_map = {
        'Title': 'title',
        'Publication Year': 'year',
        'Publication Title': 'journal',
        'ISSN': 'issn',
        'Volume': 'volume',
        'Issue': 'issue',
        'Pages': 'pages',
        'Author': 'authors',
        'Language': 'language',
        'Abstract Note': 'abstract',
        'DOI': 'doi',
        'Manual Tags': 'keywords'
    }
    
    # Rename columns
    df.rename(columns=column_map, inplace=True)
    
    # Ensure all required columns are present and lowercase
    missing_columns = set(column_map.values()) - set(df.columns.str.lower())
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Remove unwanted columns
    df = df[list(column_map.values())]
    
    return df


def flag_hollow_entries(df, config):
    """
    Flags entries with insufficient content based on non-stop words density
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    config : dict
        User configuration information
    
    Returns:
    --------
    pd.DataFrame
        The DataFrame with a new column 'hollow_flag' indicating hollow entries.
    """
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    stop_words = set(stopwords.words('english'))
    
    try:
        quantile = float(config.get('hollow_quantile', '0.15'))
    except ValueError:
        raise ValueError(f"hollow_quantile value in config file is not a float: {config.get('hollow_quantile')}")
    
    try:
        threshold = int(config.get('hollow_quantile_threshold', '5'))
    except ValueError:
        raise ValueError(f"hollow_quantile_threshold value in config file is not an integer: {config.get('hollow_quantile_threshold')}")
    
    # Calculate non-stop words density
    df['non_stop_words_density'] = df['abstract'].apply(lambda text: len([word for word in word_tokenize(text) if word.lower() not in stop_words]))
    
    # Flag hollow entries
    threshold = df['non_stop_words_density'].quantile(quantile)
    df['hollow_flag'] = df['non_stop_words_density'] < threshold
    
    return df


def assess_doi_recovery(df, config):
    """
    Assesses entries for DOI recovery and flags them for automatic or manual processing.
    
    Parameters:
    - df (DataFrame): The DataFrame containing article metadata.
    config : dict
        User configuration information
    
    Returns:
    - DataFrame: Updated DataFrame with DOI recovery flags.
    """
    
    # Isolate entries with missing DOIs
    df_filtered = df[df['doi'] == ''].copy()

    # Assess potential for DOI recovery based on presence of 'title' and 'authors'
    df_filtered.loc[:,'doi_recovery_flag'] = df_filtered.apply(
        lambda row: 'auto' if (row['title'] != '' and row['authors'] != '') else 'manual', axis=1
    )
    
    # Flag entries for manual DOI recovery if hollow
    df_filtered.loc[df_filtered.hollow_flag & (df_filtered.authors == ''), 'doi_recovery_flag'] = 'manual'
    
    # Check if doi_recovery_flag column exists in df
    if 'doi_recovery_flag' not in df.columns:
        # If it doesn't exist, create an empty column
        df['doi_recovery_flag'] = ''
    
    # Update original DataFrame with recovery flags
    df.update(df_filtered[['doi_recovery_flag']])

    # Generate CSV for manual review
    manual_review_csv = config.get('doi_manual_csv', './data/search_processing/doi_manual_review.csv')
    df_filtered['update_flag'] = ''
    cols_to_save = ['title', 'update_flag', 'authors', 'journal', 'year', 'abstract', 'doi', 'hollow_flag']
    df_filtered[df_filtered['doi_recovery_flag'] == 'manual'][cols_to_save].to_csv(manual_review_csv)
    
    print(f"Please see and review list of articles requiring manual doi review: {manual_review_csv}")
    
    # Update original DataFrame with recovery flags
    df.update(df_filtered[['doi_recovery_flag']])
    
    return df



###
#   Start 
###

# Initialize and Data Handling
config = load_user_config()
preprocessed_df, removal_log_df = initialize_preprocessing(config)

preprocessed_df = flag_hollow_entries(preprocessed_df, config)
preprocessed_df = remove_standard_text(preprocessed_df, config)

# Init dup removal
preprocessed_df, removal_log_df = remove_duplicate_dois(preprocessed_df, removal_log_df, config)

# DOI Recovery
preprocessed_df = assess_doi_recovery(preprocessed_df, config) # Flags
preprocessed_df = retrieve_missing_dois(preprocessed_df, config)

backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix='init_preproc')