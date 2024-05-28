#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 02: Data Retrieval

This script is responsible for retrieving missing data and processing manual entries. 

The process is as follows:
1. Load the preprocessed data and removal log data from the configured paths.
2. Process manual review flags using the `process_manual_entries` function. This function takes the preprocessed DataFrame, removal log DataFrame, user configuration, and the path to the manual review CSV file. It processes entries flagged for manual review and updates the 'grey_flag' column for entries marked as 'Grey Literature'.
3. Remove duplicate DOIs post data retrieval using the `remove_duplicate_dois` function.
4. Backup the current state of the data before retrieving missing data.
5. Retrieve missing data using the `retrieve_missing_data` function. This function attempts to automatically retrieve missing data for each entry. If the data retrieval is incomplete or fails, the entry is flagged for manual review. A CSV file is generated for entries requiring manual review.
6. Backup the data after automatic data retrieval.

Manual review flags:
- 'G': Grey Literature. Entries with this flag are marked as 'Grey Literature' in the 'grey_flag' column.
- 'U': Update. Entries with this flag are updated during the manual review process.
- 'R': Removal. Entries with this flag are removed during the manual review process.

Automatic data retrieval:
It queries the Crossref, PubMed, and OpenCitations APIs to retrieve metadata information
for articles based on their Digital Object Identifier (DOI). The retrieved metadata is then
used to update the entry with any available information such as title, abstract, authors,
journal, volume, issue, pages, year, and keywords.

If the retrieval process fails for any API, the entry is flagged for manual review. The script also includes functions to format and update the retrieved metadata.

NEXT STEP: User must attempt to retrieve missing data manually and update the relevant columns in the manual review data file. The user should then proceed to the next step in the preprocessing pipeline (duplicate_review.py).

Requirements:
Please review the list of articles requiring manual data retrieval and update the flags as necessary.

Configuration entries:
   - 'auto_data_retrieval_log': Path to the log file for automatic data retrieval.
   - 'manual_retrieval_csv': Path to the CSV file for manual data retrieval.
   - 'doi_manual_csv': Path to the CSV file containing DOI entries for manual review.
   - 'min_length': Minimum length for abstracts.
   - 'max_length': Maximum length for abstracts.
   - 'year_min': Minimum publication year.
   - 'year_max': Maximum publication year.

Created on Tue Apr  2 16:26:33 2024

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_preproc import *


def query_crossref(doi, logger):
    """
    Query Crossref API for metadata using DOI.

    Parameters:
    - doi (str): Digital Object Identifier (DOI) of the publication.
    logger : Logger
        Logger object for logging API requests and responses.

    Returns:
    - dict: Metadata retrieved from Crossref API.
    """
    
    logger.info(f"Querying Crossref for DOI: {doi}")
    
    crossref_url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(crossref_url)
    if response.status_code == 200:
        logger.info("Crossref metadata retrieved successfully.")
        return response.json()['message']
    else:
        logger.error(f"Crossref query failed with status code {response.status_code}.")
        
    return {}


def query_pubmed(doi, logger):
    """
    Query PubMed for article metadata using a DOI.

    Parameters
    ----------
    doi : str
        Digital Object Identifier (DOI) of the publication.
    logger : Logger
        Logger object for logging API requests and responses.

    Returns
    -------
    bs4.BeautifulSoup xml
        Metadata retrieved from PubMed
        Returns an empty dictionary if the DOI is not found or an error occurs.
    """
    logger.info(f"Querying PubMed for DOI: {doi}")
    
    # Base URL for PubMed E-utilities
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Step 1: Use esearch to find the PMID for the given DOI
    search_url = f"{base_url}/esearch.fcgi?db=pubmed&term={doi}[DOI]&retmode=json"
    search_response = requests.get(search_url)
    if search_response.status_code != 200:
        logger.error(f"PubMed search query failed with status code {search_response.status_code}.")
        return {}  # Return empty dict if the search request failed

    search_data = search_response.json()
    pmid_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not pmid_list:
        logger.info("No PMID found for given DOI.")
        return {}  # Return empty dict if no PMID found for the DOI

    pmid = pmid_list[0]  # Assuming the first PMID is the relevant one

    # Step 2: Use esummary to retrieve the article metadata using PMID
    fetch_url = f"{base_url}/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
    fetch_response = requests.get(fetch_url)
    if fetch_response.status_code != 200:
        logger.error(f"PubMed fetch query failed with status code {fetch_response.status_code}.")
        return {}

    # Parse the XML response
    soup = BeautifulSoup(fetch_response.content, 'lxml')
    
    logger.info("PubMed metadata retrieved successfully.")

    return soup


def query_occ(doi, logger):
    """
    Query OpenCitations API for metadata using DOI.

    Parameters:
    - doi (str): Digital Object Identifier (DOI) of the publication.
    logger : Logger
        Logger object for logging API requests and responses.

    Returns:
    - dict: Metadata retrieved from OpenCitations API.
    """
    
    logger.info(f"Querying OpenCitations for DOI: {doi}")
    
    occ_url = f"https://opencitations.net/index/coci/api/v1/metadata/{doi}"
    response = requests.get(occ_url)
    if response.status_code == 200:
        logger.info("OpenCitations metadata retrieved successfully.")
        return response.json()[0] if response.json() else {}
    
    logger.error(f"OpenCitations query failed with status code {response.status_code}.")
    return {}


def format_author_crossref(author):
    """Format author's name retrieved from Crossref metadata."""
    if 'given' in author:
        initials = ''.join([part[0] for part in author.get('given', '').replace('.', '').split(' ')])
        return f"{author.get('family')}, {initials}"
    else:
        return author.get('name', 'Unknown')


def update_from_crossref(entry, metadata, config):
    """
    Update entry information from Crossref metadata
    
    Parameters
    ----------
    entry : dict
        The entry to be updated with metadata information.
    metadata : dict
        Metadata information retrieved from Crossref.
    config : dict
        Configuration settings.
    
    Returns
    -------
    dict
        Updated entry with metadata information.
    """
    
    # Update the entry with any retrieved metadata
    for key in ['title', 'abstract', 'author', 'journal', 'volume', 'issue', 'page', 'year', 'keywords']:
        if key in metadata:
            if key == 'abstract':
                abstract = extract_or_return(metadata.get(key))
                abstract_clean = remove_standard_text_entry(abstract.lower(), config)
                
                # Should the abstract be replaced
                if len(abstract_clean) > 1.5*len(entry[key]):
                    entry[key] = abstract_clean
                    entry['orig_abstract'] = abstract
                continue
            elif key == 'author':
                authors = '; '.join([format_author_crossref(a) for a in metadata.get('author', [])])
                if entry['authors'] == '' or entry['authors'] != authors:
                    entry['authors'] = authors
            elif key == 'page':
                pages = extract_or_return(metadata.get('page'))
                if entry['pages'] == '' or entry['pages'] != pages:
                    entry['pages'] = pages
            elif key == 'year':
                year = metadata.get('published-print', {}).get('date-parts', [[None]])[0][0]
                year = year or metadata.get('published-online', {}).get('date-parts', [[None]])[0][0]  # Fallback to online date
                if str(entry['year']) == '' or str(int(entry['year'])) != year:
                    entry['year'] = year
            elif key == 'keywords':
                keywords = '; '.join(metadata['subject'])
                if entry['keywords'] == '':
                    entry['keywords'] = keywords
                elif  entry['keywords'] != metadata['keywords']:
                    entry['keywords'] = entry['keywords'] + '; ' + keywords.lower()
            elif entry[key] == '':
                if key == 'journal':
                    entry[key] = extract_or_return(metadata.get('container-title'))
                    continue
                
                entry[key] = extract_or_return(metadata.get(key))
    
    return entry


def update_from_pubmed(entry, xml_obj, config):
    """
    Update entry information from PubMed metadata
    
    Parameters
    ----------
    entry : dict
        The entry to be updated with metadata information.
    xml_obj : BeautifulSoup object
        Metadata information retrieved from PubMed in XML format.
    config : dict
        Configuration settings.
    
    Returns
    -------
    dict
        Updated entry with metadata information.
    """
    
    # Parse the XML response
    metadata = {
        'title':         xml_obj.find('articletitle').text if xml_obj.find('articletitle') else None,
        'orig_abstract': xml_obj.find('abstract').text if xml_obj.find('abstract') else '',
        'authors':       list(zip([ln.text for ln in xml_obj.find_all('lastname')], [ln.text for ln in xml_obj.find_all('initials')])) if xml_obj.find('lastname') else None,
        'journal':       xml_obj.find('articletitle').text if xml_obj.find('articletitle') else None,
        'volume':        xml_obj.find('volume').text if xml_obj.find('volume') else None,
        'issue':         xml_obj.find('issue').text if xml_obj.find('issue') else None,
        'pages':         xml_obj.find('medlinepgn').text if xml_obj.find('medlinepgn') else None,
        'year':          xml_obj.find('pubdate').find('year').text if xml_obj.find('pubdate').find('year') else None,
        'keywords':      '; '.join([val.text for val in xml_obj.find_all('keyword')]).lower() if xml_obj.find('keyword') else None
    }
    
    if metadata['orig_abstract'] != '':
        metadata['abstract'] = remove_standard_text_entry(metadata['orig_abstract'].lower(), config)
    else:
        metadata['abstract'] = None
        metadata['orig_abstract'] = None
    
    if metadata['authors']:
        metadata['authors'] =  '; '.join([val[0] + ' ' + val[1] for val in metadata['authors']])
    
    # Update entry
    for key in ['title', 'orig_abstract', 'authors', 'journal', 'volume', 'issue', 'pages', 'year', 'abstract']:
        entry[key] = metadata[key]
    
    # Update keywords
    if entry['keywords'] == '':
        entry['keywords'] = metadata['keywords']
    elif  entry['keywords'] != metadata['keywords']:
        entry['keywords'] = entry['keywords'] + '; ' + metadata['keywords']
    
    return entry


def format_occ_authors(authors_str):
    """
    Format authors' names from OpenCitations
    
    Parameters
    ----------
    authors_str : str
        A string containing authors' names separated by semicolons.
    
    Returns
    -------
    str
        A formatted string containing authors' names.
    
    Notes
    -----
    This function removes numerical IDs from the authors' names and formats them in the following way:
    - Last name followed by a comma
    - Initials (first letters of the first names) separated by a space
    - If no initials are available, the last name is used alone.
    """
    # Split the authors by semicolon
    authors = authors_str.split('; ')
    formatted_authors = []

    for author in authors:
        # Remove numerical IDs using regular expression, targeting those at the end or preceded by a space
        clean_author = re.sub(r'(?:^|\s)[\d\-]+$', '', author).strip()
        # Split by the last comma to separate last name and initials
        parts = clean_author.rsplit(', ', 1)
        # If the author name is in the correct format, reformat it
        if len(parts) == 2:
            last_name, initials = parts
            if initials:  # Check if initials is not empty
                formatted_author = f"{last_name}, {initials[0]}"
            else:
                formatted_author = last_name  # Use only last name if initials are empty
        else:
            # If there's no comma, try to find the first space to use as a separator for the initial
            space_index = parts[0].find(' ')
            if space_index != -1 and space_index + 1 < len(parts[0]):
                formatted_author = f"{parts[0][:space_index]}, {parts[0][space_index+1].upper()}"
            else:
                # If no space found or no character after space, use the whole name
                formatted_author = parts[0]
        formatted_authors.append(formatted_author)

    return '; '.join(formatted_authors)


def update_from_occ(entry, metadata, config):
    """
    Update entry information from OCC metadata
    
    Parameters
    ----------
    entry : dict
        The entry to be updated with metadata information.
    metadata : dict
        Metadata information retrieved from OpenCitations.
    config : dict
        Configuration settings.
    
    Returns
    -------
    dict
        Updated entry with metadata information.
    """
    
    # Update the entry with any retrieved metadata
    for key in ['title', 'abstract', 'author', 'journal', 'volume', 'issue', 'page', 'year']:
        if key in metadata:
            if key == 'abstract': # Might not be included...
                abstract = extract_or_return(metadata.get(key))
                abstract_clean = remove_standard_text_entry(abstract.lower(), config)
                
                # Should the abstract be replaced
                if len(abstract_clean) > 1.5*len(entry[key]):
                    entry[key] = abstract_clean
                    entry['orig_abstract'] = abstract
                continue
            elif key == 'author': #[x]
                authors = format_occ_authors(metadata.get('author'))
                if entry['authors'] == '' or entry['authors'] != authors:
                    entry['authors'] = authors
            elif key == 'page': #[x]
                pages = extract_or_return(metadata.get('page'))
                if entry['pages'] == '' or entry['pages'] != pages:
                    entry['pages'] = pages
            elif key == 'year':
                year = metadata.get('year').split('-')[0]
                if str(entry['year']) == '' or str(int(entry['year'])) != year:
                    entry['year'] = year
            elif entry[key] == '':
                if key == 'journal':
                    entry[key] = extract_or_return(metadata.get('source_title'))
                    continue
                
                entry[key] = extract_or_return(metadata.get(key))
    
    return entry


def update_metadata(entry, config, logger):
    """
    Update metadata information for a publication entry.

    This function attempts to retrieve metadata information from Crossref,
    OpenCitations, and PubMed APIs and updates the entry with any retrieved
    metadata.

    Parameters:
    - entry (dict): Metadata entry for a publication.
    - config (dict): Configuration settings.
    logger : Logger
        Logger object for logging API requests and responses.

    Returns:
    - dict: Updated entry with metadata information.
    """
    
    # Attempt to retrieve metadata from Crossref
    metadata = query_crossref(entry['doi'], logger)
    
    if metadata:
        return update_from_crossref(entry, metadata, config)
    
    # Attempt to retrieve metadata from PubMed if Crossref fails
    metadata = query_pubmed(entry['doi'], logger) # returns a different data type (soup xml)
    if metadata:
        return update_from_pubmed(entry, metadata, config)
    
    # Attempt to retrieve metadata from OCC if PubMed fails
    metadata = query_occ(entry['doi'], logger)
    
    if metadata:
        return update_from_occ(entry, metadata, config)
    
    # If the process fails, then flag it as manual
    entry['data_ret_flag'] = 'manual'
    return entry


def retrieve_missing_data(df, config):
    """
    Retrieve missing data for articles and flag entries for manual review.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing article metadata.
    config : dict
        Configuration settings.

    Returns
    -------
    DataFrame
        Updated DataFrame with missing data retrieved and flagged for review.
    """
    
    # Configure logging to output to both file and terminal
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler for logging to a file
    file_handler = logging.FileHandler(config.get('auto_data_retrieval_log', './data/logs/auto_data_retrieval.txt'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Stream handler for logging to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # Define key columns to check for missing data
    key_cols = ['title', 'abstract', 'authors', 'year']
    
    df['data_ret_flag'] = ''

    # Filter entries with missing data in any of the key columns and those which have a doi
    missing_data_df = df[(df[key_cols] == '').any(axis=1)]
    df.loc[missing_data_df.index, 'data_ret_flag'] = 'manual'
    missing_data_df.loc[missing_data_df.index, 'data_ret_flag'] = 'manual'
    
    missing_data_df = missing_data_df[missing_data_df.doi != '']
    df.loc[missing_data_df.index, 'data_ret_flag'] = 'auto'
    missing_data_df.loc[missing_data_df.index, 'data_ret_flag'] = 'auto'
    
    # Help track progress
    total_auto_entries = len(df[df['data_ret_flag'] == 'auto'])
    orig_manual_count  = len(df[df['data_ret_flag'] == 'manual'])
    data_found_count = 0
    new_manual_count = 0
    
    logger.info(f"Starting auto data retrieval for {total_auto_entries} articles.")
    
    # Scale logging/messaging
    max_entries_threshold = 200
    scaling_factor = 1.0 / (1.0 + total_auto_entries / max_entries_threshold)
    
    # Attempt to update metadata for each entry with missing data
    for index, row in missing_data_df.iterrows():
        updated_entry = update_metadata(row.to_dict(), config, logger)
        df.loc[index] = updated_entry
        if any(updated_entry[col] == '' for col in key_cols):
            # Flag for manual review if metadata update fails or is incomplete
            df.at[index, 'data_ret_flag'] = 'manual'
            new_manual_count += 1
        else:
            df.at[index, 'data_ret_flag'] = 'auto'
            data_found_count += 1
            
        curr_step = data_found_count + new_manual_count
        if curr_step % max(int(20 * scaling_factor), 1) == 0:
            logger.info(f"Progress: {index}: {curr_step}/{total_auto_entries}, Data found: {data_found_count}, Manual: {new_manual_count}")

    # Generate a CSV for entries requiring manual review
    manual_retrieval_df       = df[df['data_ret_flag'] == 'manual']
    manual_retrieval_csv_path = config.get('manual_retrieval_csv', './data/search_processing/manual_data_retrieval.csv')
    manual_retrieval_df['update_flag'] = ''
    manual_retrieval_df[key_cols + ['keywords', 'orig_index', 'data_ret_flag', 'doi', 'grey_flag', 'hollow_flag', 'update_flag']].to_csv(manual_retrieval_csv_path, index=False)
    
    logger.info(f"{data_found_count} out of {total_auto_entries} missing data entries resolved. {new_manual_count + orig_manual_count} entries left to review manually.")
    logger.info(f"Please see and review list of articles requiring manual data retrieval: {manual_retrieval_csv_path}")

    return df



###
#   Start 
###

# Load preprocessed data
config = load_user_config()

preprocessed_df, removal_log_df = load_data(config)

# Proces manual review flags
preprocessed_df, removal_log_df = process_manual_entries(preprocessed_df, removal_log_df, config, config.get('doi_manual_csv', './data/search_processing/doi_manual_review.csv'))

# Post DOI ret dup removal
preprocessed_df, removal_log_df = remove_duplicate_dois(preprocessed_df, removal_log_df, config, step_str = 'Post DOI Ret Duplicate Removal')

# Backup this version before retrieving
backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix='pre_data_ret')

# Retrieve data
preprocessed_df = retrieve_missing_data(preprocessed_df, config)

backup_and_save(preprocessed_df, removal_log_df, config, extra_backup_suffix='auto_data_ret')