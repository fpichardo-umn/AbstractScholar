# -*- coding: utf-8 -*-
"""
STEP 05: Process User-Categorized Articles

This script processes the user-edited CSV file of categorized articles resulting from the previous clustering and manual review steps. It separates the articles into two categories based on user judgment: those to be kept for further analysis and those to be dropped. The script can operate in two modes, determined by a configuration setting, to either prepare data for reprocessing through the entire pipeline or to continue with the next step in the existing pipeline.

The script performs the following main tasks:
1. Loads the user-categorized data from a CSV file.
2. Processes and saves entries marked for removal.
3. Processes and saves entries marked to be kept, with the output format depending on the configuration.
4. Provides informative console output about the processing results.

The script outputs two files:
1. A CSV file containing the articles to be kept, with columns determined by the processing mode.
2. A CSV file logging the removed articles and the reason for removal.

Description of Method:
1. **Data Loading**: The script loads the user-categorized data from a CSV file specified in the configuration.
2. **Dropped Entries Processing**: Entries marked as 'D' (drop) by the user are separated and saved to a removal log file with the reason "Irrelevant".
3. **Kept Entries Processing**: Entries marked as 'K' (keep) by the user are processed in one of two ways:
   a. If 'reset_for_pipeline' is True: Only specified columns are kept, preparing the data for reprocessing through the entire pipeline.
   b. If 'reset_for_pipeline' is False: All columns except 'user_judgment' are kept, preparing the data for the next step in the existing pipeline.
4. **Output**: The processed kept entries are saved to a CSV file, with the filename and included columns determined by the 'reset_for_pipeline' setting.

The script uses a configuration file to determine file paths and processing mode, allowing for flexibility in its operation.

Created on 24 July 2024
@author: Felix Pichardo and Claude
"""

import pandas as pd
import os
import os.path as op
import sys
import glob
import shutil
from typing import List

# Add the 'scripts' subdirectory to the Python path
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_text import *

# Constants
COLUMNS_TO_KEEP = [
    'title', 'year', 'journal', 'issn', 'volume', 'issue', 'pages', 'authors',
    'language', 'abstract', 'doi', 'keywords', 'orig_abstract',
    'non_stop_words_density', 'hollow_flag', 'doi_recovery_flag', 'orig_index',
    'grey_flag', 'data_ret_flag'
]

def load_categorized_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

def process_dropped_entries(data: pd.DataFrame, output_file: str):
    """Process and save dropped entries."""
    dropped = data[data['user_judgment'] == 'D'].copy()
    dropped['removal_reason'] = 'Irrelevant'
    
    if op.exists(output_file):
        orig_dropped = pd.read_csv(output_file)
        dropped = pd.concat([orig_dropped, dropped])
    
    dropped.to_csv(output_file, index=False)
    print(f"Dropped {len(dropped)} entries. Saved to {output_file}")

def version_preprocessed_data(config: dict):
    """Version the existing preprocessed_data.csv file."""
    preprocessed_file = config['preprocessed_data_file']
    base_path = os.path.dirname(preprocessed_file)
    
    if not os.path.exists(preprocessed_file):
        print(f"Warning: {preprocessed_file} does not exist. No versioning performed.")
        return

    # Count existing versioned files
    version_count = len(glob.glob(os.path.join(base_path, 'preprocessed_data_v*.csv')))
    new_version = version_count + 1
    
    # Copy the current file to the new versioned name
    new_filename = os.path.join(base_path, f'preprocessed_data_v{new_version}.csv')
    shutil.copy2(preprocessed_file, new_filename)
    print(f"Existing preprocessed_data.csv copied to {new_filename}")

def process_kept_entries(data: pd.DataFrame, reset_for_pipeline: bool, columns_to_keep: List[str], config: dict):
    """Process and save kept entries based on reset_for_pipeline configuration."""
    kept = data[data['user_judgment'] == 'K'].copy()
    
    if reset_for_pipeline:
        # Version the existing preprocessed_data.csv
        version_preprocessed_data(config)
        
        # Keep only specified columns for reprocessing
        missing_columns = set(columns_to_keep) - set(kept.columns)
        if missing_columns:
            print(f"Warning: The following columns are missing: {missing_columns}")
            columns_to_keep = [col for col in columns_to_keep if col not in missing_columns]
        kept = kept[columns_to_keep]
        
        # Save to preprocessed_data.csv (overwriting the original)
        output_file = config['preprocessed_data_file']
    else:
        # Keep all columns except 'user_judgment' for continuing the pipeline
        kept = kept.drop('user_judgment', axis=1)
        output_file = config['post_review_file']
    
    kept.to_csv(output_file, index=False)
    print(f"Kept {len(kept)} entries. Saved to {output_file}")

# Load configuration
config = load_user_config()

# Load categorized data
categorized_data_file = normalize_path(config.get('categorized_data_file', 'data/text_analysis/categorized_articles.csv'))
categorized_data = load_categorized_data(categorized_data_file)

# Process dropped entries
removal_log_file = normalize_path(config.get('removal_log_file', 'data/text_removal_log.csv'))
process_dropped_entries(categorized_data, removal_log_file)

# Determine processing based on reset_for_pipeline
reset_for_pipeline = config.get('reset_for_pipeline', 'false').lower() in ('true', 'yes', '1', 'on')

# Ensure necessary paths are in the config
config['preprocessed_data_file'] = normalize_path(config.get('preprocessed_data_file', 'data/preprocessed_data.csv'))
config['post_review_file'] = normalize_path(config.get('post_review_file', 'data/text/text_post_abstract_review.csv'))

# Process kept entries
process_kept_entries(categorized_data, reset_for_pipeline, COLUMNS_TO_KEEP, config)

print("Processing complete.")