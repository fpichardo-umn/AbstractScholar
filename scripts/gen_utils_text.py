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
import shutil
import logging
import json
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import string



def display_topics_tab(model, feature_names, num_top_words):
    """
    Display the top words for each topic in a given model.

    Args:
        model (object): The topic model object.
        feature_names (list): List of feature names.
        num_top_words (int): Number of top words to display for each topic.
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))
        input()


def get_article_top_concepts(docs, concepts_transformed_data):
    """Get the top concepts for each article.

    This function takes in a list of documents and a matrix of transformed data
    representing the concepts. It calculates the top concepts for each article
    based on the transformed data.

    Args:
        docs (list): A list of documents.
        concepts_transformed_data (numpy.ndarray): A matrix of transformed data
            representing the concepts.

    Returns:
        dict: A dictionary where the keys are article indices and the values are
            lists of the top concepts for each article.
    """

    art_top_concepts = {art:[] for art in range(len(docs))}
    top_num = 5
    for i in range(len(concepts_transformed_data)):
       top_topics = np.argsort(concepts_transformed_data[i,:])[::-1][:top_num]
       top_topics_str = ' '.join(str(t) for t in top_topics)
       art_top_concepts[i] = [int(top) for top in top_topics_str.split()]

    return art_top_concepts


def get_terms_for_concept(terms, concept_matrix, concept_idx, top_n=10):
    """
    Get the top n terms for a given concept in a concept matrix.

    Parameters:
    terms (list): A list of terms.
    concept_matrix (numpy.ndarray): A concept matrix.
    concept_idx (int): The index of the concept.
    top_n (int, optional): The number of top terms to retrieve. Defaults to 10.

    Returns:
    list: The top n terms for the given concept.
    """
    concept_scores = concept_matrix[concept_idx]
    top_indices = np.argsort(concept_scores)[::-1][:top_n]
    top_terms = [terms[i] for i in top_indices]

    return top_terms


def get_top_words_art(art_idx_list, docs, terms, components, doc_term_mat_xfm, concepts_per_art=3, terms_per_concept=10):
    """
    Retrieves the top words for a given list of articles.

    Args:
        art_idx_list (list): A list of article indices.
        docs (list): A list of documents.
        terms (list): A list of terms.
        components (list): A list of components.
        doc_term_mat_xfm (numpy.ndarray): A matrix representing the document-term matrix transformed by some method.
        concepts_per_art (int, optional): The number of concepts per article. Defaults to 3.
        terms_per_concept (int, optional): The number of terms per concept. Defaults to 10.

    Returns:
        list: A list of top terms for each concept.
    """
    if not 'article_top_concepts' in dir():
        article_top_concepts = get_article_top_concepts(docs, doc_term_mat_xfm)

    concepts = []
    for art in art_idx_list:
        concepts.extend(article_top_concepts[art][:concepts_per_art])

    concepts = list(set(concepts))

    top_terms = []
    for concept in concepts:
        top_terms.append(get_terms_for_concept(terms, components, concept, terms_per_concept))

    return top_terms


def load_user_config(config_file='user_config.txt'):
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            config[key.strip()] = value.strip()
    return config


def load_data(config):
    """
    Load preprocessed data.

    Parameters:
    -----------
    config : dict
        Configuration parameters.

    Returns:
    --------
    pandas DataFrames containing updated preprocessed data DataFrame
    """
    # Define paths for preprocessed data
    preprocessed_path = config.get('preprocessed_file_path', './data/preprocessed_data.csv')

    # Check if files exist
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError("The preprocessed data file does not exist.")

    # Load preprocessed data
    preprocessed_df = pd.read_csv(preprocessed_path, index_col='orig_index', encoding='latin1')
    
    preprocessed_df.index.name = 'index'
    preprocessed_df['orig_index'] = preprocessed_df.index
    preprocessed_df = replace_nulls(preprocessed_df, preprocessed_df.columns)
    preprocessed_df[preprocessed_df.select_dtypes(exclude=['object']).columns] = preprocessed_df[preprocessed_df.select_dtypes(exclude=['object']).columns].astype(int).astype(str)
    preprocessed_df.index = preprocessed_df.index.astype(int).astype(str)
    
    return preprocessed_df


def replace_nulls(df, columns):
    """Replace null values with empty strings in specified columns"""
    for col in columns:
        df[col] = df[col].fillna('')
    return df


def save_to_pickle(data, file_path):
    """
    Save the specified data to a pickle file at the given file path.

    Args:
    data (list): The data to be serialized and saved.
    file_path (str): The path where the pickle file should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}. Error: {e}")






def backup_and_save2222(preprocessed_df, removal_log_df, config, extra_backup_suffix=None):
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