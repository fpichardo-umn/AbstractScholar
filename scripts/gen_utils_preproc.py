#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script performs the following tasks:
- Imports necessary libraries and modules
- Defines helper functions for data cleaning, processing, and manipulation
- Implements functions for retrieving DOIs from different sources
- Handles cases where a DOI cannot be found automatically and flags them for manual review
- Saves the updated dataset and removal log

Created on Tue Apr  2 09:06:19 2024

@author: Felix Pichardo
"""

import os
import re
import requests
import shutil
import logging
import os.path as op
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from urllib.parse import quote_plus, quote
from fuzzywuzzy import fuzz # requires python-Levenshtein


def replace_nulls(df, columns):
    """Replace null values with empty strings in specified columns"""
    for col in columns:
        df[col] = df[col].fillna('')
    return df


def extract_or_return(data):
    """Check if the input data is a list. If it is, return its first element; otherwise, return the data itself"""
    if type(data) == list:
        return data[0]
    else:
        return data


def load_user_config(config_file='user_config.txt'):
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            config[key.strip()] = value.strip()
    return config


def update_doi_counts(removal_log_df, preprocessed_df):
    """
    Update DOI counts in the removal log DataFrame.

    Parameters
    ----------
    removal_log_df : DataFrame
        DataFrame containing removal log information.
    preprocessed_df : DataFrame
        DataFrame containing preprocessed article metadata.

    Returns
    -------
    DataFrame
        Updated removal log DataFrame with DOI counts.
    """
    # Compile a comprehensive list of DOIs
    all_dois = pd.concat([removal_log_df['doi'], preprocessed_df['doi']])
    
    # Calculate updated counts for all DOIs
    all_doi_counts = all_dois.value_counts().to_dict()
    
    # Update 'dup_doi_cnt' in the removal log
    if 'dup_doi_cnt' in removal_log_df.columns:
        removal_log_df['dup_doi_cnt'] = removal_log_df['doi'].map(all_doi_counts)
    else:
        # If 'dup_doi_cnt' column doesn't exist, create it and populate
        removal_log_df = removal_log_df.assign(dup_doi_cnt=removal_log_df['doi'].map(all_doi_counts))
    
    return removal_log_df


def remove_duplicate_dois(preprocessed_df, removal_log_df, config, step_str = 'Initial DOI Duplicate Removal'):
    """
    Remove duplicate DOIs from the preprocessed data.

    Parameters:
    -----------
    preprocessed_df : pandas DataFrame
        DataFrame containing preprocessed data.
    removal_log_df : pandas DataFrame
        DataFrame to log removal operations.
    config : dict
        Configuration parameters
    step_str : str (optional)
        Step of removal to append to log
    """
    
    # Identify duplicate DOIs - and ignore the blank ones
    duplicate_dois = preprocessed_df[preprocessed_df['doi'].duplicated() & (preprocessed_df['doi'] != '')]
    
    # Add new duplicate entries to the removal log, initializing 'dup_doi_cnt'
    new_removals   = duplicate_dois.assign(reason_for_removal='Duplicate DOI', removal_step=step_str, dup_doi_cnt=0)
    removal_log_df = pd.concat([removal_log_df, new_removals])

    # Recalculate 'dup_doi_cnt' for all DOIs
    all_dois = pd.concat([removal_log_df['doi'], duplicate_dois['doi']]).replace('', np.nan)
    all_dois = all_dois.dropna()
    all_doi_counts = all_dois.value_counts().to_dict()
    removal_log_df['dup_doi_cnt'] = removal_log_df['doi'].map(all_doi_counts)

    # Remove duplicates from the preprocessed data
    preprocessed_df.drop(duplicate_dois.index, inplace=True)
    
    # Backup and save changes
    backup_and_save(preprocessed_df, removal_log_df, config)
    
    return preprocessed_df, removal_log_df


def backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix=None):
    """
    Backup current data and save changes, with an option for an additional backup using a suffix.

    Parameters:
    -----------
    preprocessed_df : pandas DataFrame
        DataFrame containing preprocessed data.
    removal_log_df : pandas DataFrame
        DataFrame containing removal log.
    config : dict
        Configuration parameters.
    extra_backup_suffix : str, optional
        Suffix to add for an additional backup, before the file extension.
    """
    
    # Define current file paths
    preprocessed_path = config.get('preprocessed_file_path', './data/preprocessed_data.csv')
    removal_log_path = config.get('removal_log_path', './data/search_processing/removal_log.csv')

    # Define backup paths from config
    backup_dir = config.get('backup_dir', './data/backup/')
    preprocessed_backup_path = os.path.join(backup_dir, 'preprocessed_backup.csv')
    removal_log_backup_path = os.path.join(backup_dir, 'removal_log_backup.csv')
    
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup current files by copying them
    shutil.copy(preprocessed_path, preprocessed_backup_path)
    shutil.copy(removal_log_path, removal_log_backup_path)
    
    # Save updated DataFrames to original paths
    preprocessed_df.to_csv(preprocessed_path, index=False)
    removal_log_df.to_csv(removal_log_path, index=False)
    
    # Perform extra backup with suffix, if provided
    if extra_backup_suffix:
        extra_preprocessed_backup_path = preprocessed_backup_path.replace('.csv', f'_{extra_backup_suffix}.csv')
        extra_removal_log_backup_path = removal_log_backup_path.replace('.csv', f'_{extra_backup_suffix}.csv')
        shutil.copy(preprocessed_path, extra_preprocessed_backup_path)
        shutil.copy(removal_log_path, extra_removal_log_backup_path)
    

def load_data(config):
    """
    Load preprocessed data.

    Parameters:
    -----------
    config : dict
        Configuration parameters.

    Returns:
    --------
    tuple of pandas DataFrames
        Tuple containing updated preprocessed data DataFrame
        and removal log DataFrame.
    """
    # Define paths for preprocessed data and removal log
    preprocessed_path = config.get('preprocessed_file_path', './data/preprocessed_data.csv')
    removal_log_path  = config.get('removal_log_path', './data/search_processing/removal_log.csv')

    # Check if files exist
    if not (os.path.exists(preprocessed_path) and os.path.exists(removal_log_path)):
        raise FileNotFoundError("One or both of the specified files do not exist.")

    # Load preprocessed data and removal log
    preprocessed_df = pd.read_csv(preprocessed_path, index_col='orig_index', encoding='latin1')
    removal_log_df  = pd.read_csv(removal_log_path, index_col='orig_index', encoding='latin1')
    
    preprocessed_df.index.name = 'index'
    removal_log_df.index.name = 'index'
    
    preprocessed_df['orig_index'] = preprocessed_df.index
    removal_log_df['orig_index']  = removal_log_df.index
    
    preprocessed_df = replace_nulls(preprocessed_df, preprocessed_df.columns)
    removal_log_df  = replace_nulls(removal_log_df, removal_log_df.columns)
    
    preprocessed_df[preprocessed_df.select_dtypes(exclude=['object']).columns] = preprocessed_df[preprocessed_df.select_dtypes(exclude=['object']).columns].astype(int).astype(str)
    removal_log_df[removal_log_df.select_dtypes(exclude=['object']).columns] = removal_log_df[removal_log_df.select_dtypes(exclude=['object']).columns].astype(int).astype(str)
    
    preprocessed_df.index = preprocessed_df.index.astype(int).astype(str)
    removal_log_df.index = removal_log_df.index.astype(int).astype(str)
    
    return preprocessed_df, removal_log_df


def remove_long_standard_text(df, config):
    """
    Removes repetitive, contentless blocks of text from the 'abstract' column
    It uses a sliding window to find common text sequences and removes them if they exceed a defined repetition threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process
    config : dict
        User configuration information
    
    Returns:
    --------
    pd.DataFrame
        The DataFrame with cleaned 'abstract' column
    """
    
    from collections import Counter
    
    try:
        window_size = int(config.get('std_txt_window_size', '5'))
    except ValueError:
        raise ValueError(f"std_txt_window_size value in config file is not a integer: {config.get('std_txt_window_size')}")
    
    try:
        repetition_threshold = int(config.get('std_txt_rep_thresh', '5'))
    except ValueError:
        raise ValueError(f"std_txt_rep_thresh value in config file is not an integer: {config.get('std_txt_rep_thresh')}")
        
    ignore_std_txt = config.get('ignore_std_txt_str', '').split(',')
    term_removal_log_path = config.get('term_removal_log_path', './data/search_processing/term_removal_log.csv')
    
    term_counts = Counter()  # To track terms exceeding the threshold
    
    # Identify and remove repetitive sequences
    for index, row in df.iterrows():
        tokens = row['abstract'].split()
        sequences = [' '.join(tokens[i:i+window_size]) for i in range(len(tokens) - window_size + 1)]
        sequence_counts = Counter(sequences)
        
        for seq, count in sequence_counts.items():
            if count > repetition_threshold and seq not in ignore_std_txt:
                term_counts[seq] += count
                df.at[index, 'abstract'] = df.at[index, 'abstract'].replace(seq, '')
    
    # Export term removal log if there are terms that exceed the threshold
    if term_counts:
        term_removal_df = pd.DataFrame(list(term_counts.items()), columns=['Removed Term', 'Count'])
        term_removal_df.to_csv(term_removal_log_path, index=False)
    
    return df


def remove_copyright_notices(abstract):
    """Remove copyright notices and any text following them from the abstract"""
    
    # Remove (c) followed by a year (with or without space) and anything after
    abstract = re.sub(r'\(c\)\s*(?:20\d{2}|19\d{2}).*', '', abstract, flags=re.IGNORECASE)
    
    # Remove (c) followed by any text that eventually contains 'press' and anything after
    abstract = re.sub(r'\(c\).*?press.*', '', abstract, flags=re.IGNORECASE)
    
    # Remove other forms of copyright notice and anything after
    abstract = re.sub(r'copyright .*', '', abstract, flags=re.IGNORECASE)
    
    # Remove other forms of copyright notice and anything after
    abstract = re.sub(r'© .*', '', abstract, flags=re.IGNORECASE)
    
    # Remove database record notices and anything after
    abstract = re.sub(r'\(\w+ database record.*', '', abstract, flags=re.IGNORECASE)
    
    return abstract


def clean_abstract(abstract, header_patterns, markdown):
    """
    Cleans the given abstract text by removing HTML tags, markdown formatting, predefined header keywords, and copyright stuff
   
    Parameters:
    -----------
    abstract : str
        The abstract text to be cleaned.
    header_patterns : list
        A list of compiled regex patterns representing headers to remove (e.g., 'Introduction:', 'Background:').
    markdown : bool
        Indicates whether markdown removal should be performed on the abstract text.
    
    Returns:
    --------
    str
        The cleaned abstract text, free from HTML, markdown, and unnecessary headers.
    """
    
    # Remove HTML tags
    soup = BeautifulSoup(abstract, "html.parser")
    cleaned_text = soup.get_text(separator=" ")

    # Remove markdown
    if markdown:
        # Basic markdown patterns for bold, italic, links, etc.
        markdown_patterns = [r'\*\*(.*?)\*\*', r'__(.*?)__', r'\*(.*?)\*', r'_(.*?)_', r'!\[(.*?)\]\((.*?)\)', r'\[(.*?)\]\((.*?)\)']
        for pattern in markdown_patterns:
            cleaned_text = re.sub(pattern, r'\1', cleaned_text)

    # Remove standard headers
    for pattern in header_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)

    return remove_copyright_notices(cleaned_text)


def remove_standard_text(df, config):
    """
    Cleans the 'abstract' column in the DataFrame by removing contentless text including standard headers, HTML/markdown content, and longer standard text such as copyright notices
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with an 'abstract' column to clean.
    config : dict
        User configuration information
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned 'abstract' column.
    """
    
    header_list = ['background[:.]', 'background.[:.]', 'background and aims[:.]', 'background/aims[:.]','background and purpose[:.]','background and objectives[:.]','background & objectives[:.]',  'summary[:.]', 'introduction[:.]', 'methods[:.]', 'results[:.]', 'conclusions[:.]', 'objective[:.]', 'objectives[:.]', 'discussion[:.]', 'aims[:.]', 'aims/hypothesis', 'rationale[:.]', 'importance[:.]', '\[abstract from author\]']
    header_patterns = [re.compile(pat) for pat in header_list]

    # if markdown is used in abstracts, set markdown=True
    df['abstract'] = df['abstract'].apply(lambda x: clean_abstract(x, header_patterns, markdown=config.get('markdown', False)))
    df.abstract    = df.abstract.str.strip()
    
    # Remove longer standard text, e.g. copyright notices
    df = remove_long_standard_text(df, config)

    return df


def remove_standard_text_entry(entry, config):
    """
    Cleans the 'abstract' entry by removing contentless text including standard headers and HTML/markdown content
    
    Parameters:
    -----------
    entry : str
        Abstract as a str
    config : dict
        User configuration information
    
    Returns:
    --------
    str
        Cleaned up abstract text
    """
    
    header_list = ['background[:.]', 'background.[:.]', 'background and aims[:.]', 'background/aims[:.]','background and purpose[:.]','background and objectives[:.]','background & objectives[:.]',  'summary[:.]', 'introduction[:.]', 'methods[:.]', 'results[:.]', 'conclusions[:.]', 'objective[:.]', 'objectives[:.]', 'discussion[:.]', 'aims[:.]', 'aims/hypothesis', 'rationale[:.]', 'importance[:.]', '\[abstract from author\]']
    header_patterns = [re.compile(pat) for pat in header_list]

    # if markdown is used in abstracts, set markdown=True
    entry = clean_abstract(entry, header_patterns, markdown=config.get('markdown', False))

    return entry


def process_manual_entries(preprocessed_df, removal_log_df, config, filepath):
    """
    Process entries based on manual review outcomes and update preprocessed and removal log DataFrames.

    Parameters:
    preprocessed_df : DataFrame
        Preprocessed DataFrame to be updated.
    removal_log_df : DataFrame
        DataFrame for logging removals.
    config : dict
        Configuration settings.
    filepath : str
        path to the review file being processed

    Returns:
    Updated preprocessed_df and removal_log_df.
    """
    # Load the manual review DataFrame
    review_df  = pd.read_csv(filepath, index_col='orig_index', encoding='latin1')
    review_df  = replace_nulls(review_df, review_df.columns)
    review_df.index.name = 'index'
    review_df['orig_index'] = review_df.index

    # Update entries marked 'U' and 'G' for update
    update_df = review_df[(review_df['update_flag'] == 'U') | (review_df['update_flag'] == 'G')]
    common_columns = preprocessed_df.columns.intersection(update_df.columns)
    preprocessed_df.update(update_df[common_columns])

    # Remove entries marked 'R' for remove
    remove_df = review_df[review_df['update_flag'] == 'R']
    common_columns = preprocessed_df.columns.intersection(removal_log_df.columns)
    removal_log_df = pd.concat([removal_log_df, preprocessed_df[common_columns].loc[remove_df.index].assign(reason_for_removal='REVIEW', removal_step='Manual DOI Update')])
    preprocessed_df.drop(remove_df.index, inplace=True)

    # Mark entries as grey literature if marked 'G'
    grey_lit_df = review_df[review_df['update_flag'] == 'G']
    
    if not 'grey_flag' in preprocessed_df.columns:
        preprocessed_df['grey_flag'] = ''
    
    if grey_lit_df.shape[0] > 0:
        preprocessed_df['grey_flag'] = preprocessed_df['grey_flag'].astype(bool)
        preprocessed_df.loc[grey_lit_df.index, 'grey_flag'] = True

    # Backup and save the updated DataFrames
    backup_and_save(preprocessed_df, removal_log_df, config)

    return preprocessed_df, removal_log_df


def match_title(input_title, candidate_title, config):
    """
    Use fuzzy matching to compare titles.
    """
    # Use fuzzy matching to compare titles; consider a match if the score is above a threshold
    similarity_score = fuzz.partial_ratio(input_title.lower(), candidate_title.lower())
    return similarity_score > int(config.get('title_match_threshold', 80))


def match_authors(input_authors, candidate_authors, config):
    """
    Convert author lists to string and use fuzzy matching to compare
    """
    # Convert author lists to string
    input_authors_str     = ' '.join(input_authors.split(';')).strip()
    candidate_authors_str = ' '.join(candidate_authors).strip()
    
    # Use fuzzy matching to compare titles; consider a match if the score is above a threshold
    similarity_score = fuzz.partial_ratio(input_authors_str.lower(), candidate_authors_str.lower())
    
    return similarity_score > int(config.get('author_match_threshold', 60))


def query_crossref(title, logger):
    """
    Query Crossref API for metadata using the title

    Parameters:
    title : str
        The title of the article to search for encoded for the URL.
    logger : Logger
        Logger object for logging API requests and responses.

    Returns
    -------
    list of dict
        List of dictionaries containing metadata retrieved from Crossref API.
    """
    
    results = []
    
    # Query CrossRef API for articles by title (first encoding attempt)
    encoded_title = quote(title)
    response = requests.get(f"https://api.crossref.org/works?query.title={encoded_title}&rows=5")
    if response.status_code != 200:
        logger.info(f"CrossRef API request failed for title: {title}")
        results.append(response.json().get('message', {}).get('items', []))
    
    # Query CrossRef API for articles by title (second encoding attempt)
    encoded_title = quote_plus(title)
    response = requests.get(f"https://api.crossref.org/works?query.title={encoded_title}&rows=5")
    if response.status_code != 200:
        logger.info(f"CrossRef API request failed for title: {title}")
        results.append(response.json().get('message', {}).get('items', []))
    
    if not results:
        return None # Return None if there's an error with the API request
    
    return results


def extract_confirm_doi_crossref(title, year, authors, results, logger, config):
    """
    Extract and confirm DOI from CrossRef metadata.

    Parameters
    ----------
    title : str
        The title of the article.
    year : int
        The publication year of the article.
    authors : str
        The authors of the article.
    results : list of dict
        List of dictionaries containing metadata retrieved from Crossref API.
    logger : Logger
        Logger object for logging API requests and responses.
    config : dict
        User configuration information.

    Returns
    -------
    str or None
        The DOI of the matched article, if found; otherwise, None.
    """
    
    for works in results:
        for work in works:
            work_title = work.get('title', [''])[0]  # Titles are in a list; get the first title
            work_year = work.get('published-print', {}).get('date-parts', [[None]])[0][0]
            work_year = work_year or work.get('published-online', {}).get('date-parts', [[None]])[0][0]  # Fallback to online date
            work_authors = [f"{a.get('family')}, {a.get('given')}" for a in work.get('author', [])]
            
            # Check for similarity between years
            year_similarity_bool = fuzz.partial_ratio(str(year), str(work_year)) == 75 # off by one digit
            
            # Check for title and authors match
            if year_similarity_bool and match_title(title, work_title, config) and match_authors(authors, work_authors, config):
                logger.info(f"Match found: {work.get('DOI', 'No DOI available')}")
                return work.get('DOI', None)  # Return DOI if found
            else:
                logger.info(f"No match found for title: {title}, checking next result.")
    
    logger.info(f"No DOI found for title: {title} after checking all results.")
    
    return None  # Return None if no match is found


def search_for_article_crossref(title, year, authors, logger, config):
    """
    Search CrossRef API for an article by title, year, and authors.

    Parameters
    ----------
    title : str
        The title of the article to search for.
    year : int
        The publication year of the article.
    authors : str
        The authors of the article.
    logger : Logger
        Logger object for logging API requests and responses.
    config : dict
        User configuration information.

    Returns
    -------
    str or None
        The DOI of the matched article, if found; otherwise, None.
    """
    
    # Query CrossRef API for articles by title    
    works = query_crossref(title, title)
    
    if not works:
        return None # Return None if there's an error with the API request
    
    return extract_confirm_doi_crossref(title, year, authors, works, logger, config)


def retrieve_missing_dois(preprocessed_df, config):
    """
    Attempt to retrieve missing DOIs for articles flagged as 'auto' for DOI recovery.
    
    Parameters:
    preprocessed_df (DataFrame): The DataFrame containing article metadata.
    config (dict): User configuration information.
    
    Returns:
    DataFrame: The DataFrame with updated DOIs where found.
    """
    
    # Configure logging to output to both file and terminal
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler for logging to a file
    file_handler = logging.FileHandler(config.get('auto_retrieval_log', './data/logs/auto_retrieval.txt'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Stream handler for logging to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    manual_review_csv = config.get('doi_manual_csv', './data/search_processing/doi_manual_review.csv')
    cols_to_save = ['title', 'authors', 'journal', 'year', 'abstract', 'doi', 'hollow_flag']
    
    # Filter DataFrame to include only articles flagged as 'auto' for DOI recovery
    auto_recovery_df = preprocessed_df[preprocessed_df['doi_recovery_flag'] == 'auto']
    
    # Help track progress
    total_entries = len(preprocessed_df[preprocessed_df['doi_recovery_flag'] == 'auto'])
    doi_found_count = 0
    manual_count = 0
    
    # Initialize an empty list to hold rows that need manual review
    manual_review_entries = []
    
    logger.info(f"Starting DOI retrieval for {total_entries} articles.")
    
    for index, row in auto_recovery_df.iterrows():
        logger.info(f"Processing article index {index}: Title - {row['title']}, Year - {row['year']}, Authors - {row['authors']}")
        doi = search_for_article_crossref(row['title'], int(row['year']), row['authors'], logger, config)
        if doi:
            doi_found_count += 1
            # logging.info(f"DOI found: {doi}")
            preprocessed_df.at[index, 'doi'] = doi
        else:
            manual_count += 1
            # logging.info("No DOI found, setting to manual retrieval.")
            # Flag for manual recovery if DOI not found
            preprocessed_df.at[index, 'doi_recovery_flag'] = 'manual'
            # Add this row to the list for manual review
            manual_review_entries.append(preprocessed_df.loc[index][cols_to_save])
        curr_step = doi_found_count + manual_count
        if curr_step % 10 == 0:
            logger.info(f"Progress: {index}: {curr_step}/{total_entries}, DOIs found: {doi_found_count}, Manual: {manual_count}")

    
    logger.info("DOI retrieval completed.")
    
    # Convert the list to a DataFrame and append it to the manual_review_csv in one go
    if manual_review_entries:
        pd.DataFrame(manual_review_entries).to_csv(manual_review_csv, mode='a', header=not os.path.exists(manual_review_csv))
    

    return preprocessed_df