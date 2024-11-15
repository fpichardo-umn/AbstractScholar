![Project Image](imgs/AbScholar.png)
# Abstract Scholar

A systematic review assistant that combines automated computational methods with structured manual review processes to streamline literature screening. It helps researchers process large volumes of search results by automatically retrieving metadata, identifying duplicates, organizing articles by topic, and suggesting relevance - while maintaining expert oversight through integrated review steps.

## Key Features

- **Smart Data Retrieval**: Automated metadata collection from CrossRef, PubMed, and OpenCitations APIs
- **Intelligent Duplicate Detection**: Combines DOI matching, LSH, and content similarity analysis
- **Topic-Based Organization**: Groups similar articles using LSA and adaptive clustering
- **ML-Assisted Relevance Sorting**: Uses ensemble classification to prioritize articles
- **Structured Review Process**: Clear manual review steps with guided CSV templates
- **Comprehensive Logging**: Tracks all automated decisions and manual reviews

## Pipeline Overview

### Search Results Processing
**Primary Objective**: Obtain complete metadata and remove duplicates efficiently

- Automated DOI and metadata retrieval from multiple APIs
- Smart duplicate detection using multiple similarity measures
- Manual review integration for uncertain cases
- Comprehensive removal logging

### Text Analysis
**Primary Objective**: Organize articles into concept groups for efficient review

- LSA-based preprocessing to capture semantic relationships
- Adaptive clustering to identify coherent topic groups
- Cluster quality assessment and optional merging
- Manual review of cluster relevance

### Relevance Sorting
**Primary Objective**: Prioritize articles for detailed review

- Uses cluster assignments and ML predictions
- Ensemble classification combining multiple algorithms
- Coherence-based scoring adjustments
- Manual validation of categorizations

## Manual Review Integration

The pipeline integrates expert review at critical decision points:

1. **DOI Review**
   - Review automatically retrieved DOIs
   - Flag grey literature
   - Mark entries for removal/update

2. **Duplicate Resolution**
   - Confirm potential duplicate pairs
   - Choose primary entry for merges
   - Handle special cases

3. **Cluster Review**
   - Evaluate topic coherence
   - Mark cluster relevance
   - Guide ML training

4. **Final Article Review**
   - Validate relevance predictions
   - Make final inclusion decisions
   - Document removal reasons

## Prerequisites

- Zotero citation manager
- Initial dataset exported from Zotero containing:
  - Title
  - Publication Year
  - Journal
  - ISSN
  - Volume
  - Issue
  - Pages
  - Authors
  - Language
  - Abstract
  - DOI
  - Keywords
- Python 3.8+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fpichardo-umn/AbstractScholar.git
cd AbstractScholar
```

2. Set up a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install requirements:
```bash
# Option 1: Using pip directly
pip install -r requirements.txt

# Option 2: Using the installation script (includes NLTK data)
python install_requirements.py
```

Note: The installation script will automatically download required NLTK data. If installing manually, you'll need to run:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

## Pipeline Usage

### Configuration

Edit `user_config.txt` to configure key aspects of the pipeline:

**File Paths**
- Input/output locations for search results, preprocessed data, and review files
- Log file locations
- Backup and temporary file locations

**Data Processing**
- Language settings (e.g., "english, spanish")
- Publication year range (default: 1900-2024)
- Abstract length limits
- Custom stopwords for text analysis

**Similarity Detection**
- Title/author matching thresholds
- LSH parameters for duplicate detection
- Merge thresholds for similar entries

**Text Analysis**
- N-gram range for text processing
- Document frequency limits
- SVD component settings
- Clustering parameters
  - Minimum cluster size
  - Similarity metrics
  - Number of iterations

**Optional Features**
- Search query integration (`use_search_query`)
- Pipeline reset options
- Markdown processing

A template configuration file is provided with recommended default values. Most users will only need to modify file paths and language settings.

### Search Processing Pipeline
```bash
# Initial data preparation
python initialize_preproc.py
--> Review missing DOIs in doi_manual_review.csv
    Mark entries as:
    - Update (U): DOI found
    - Remove (R): Invalid entry
    - Grey (G): Grey literature

# Retrieve missing metadata
python data_retrieval.py
--> Review entries in manual_data_retrieval.csv
    Provide missing information for incomplete entries

# Flag potential duplicates
python duplicate_review.py
--> Review flagged entries in duplicates_review.csv
    Mark entries as:
    - Keep (K): Not a duplicate
    - Remove (R): Duplicate to remove
    - Merge (M): Merge information

# Process duplicate decisions
python duplicate_resolution.py

# Quality control checks
python final_qc_preproc.py
--> Review flagged entries in preprocessed_data.csv
```

### Text Analysis Pipeline
```bash
# Text preprocessing
python preprocess_abstracts.py

# Topic clustering
python cluster_graph.py

# Define topic groups
python define_clusters.py
--> Review clusters in text_review_clusters.csv
    Mark clusters as:
    - Relevant (R)
    - Irrelevant (I)
    - Borderline (B)

# Train relevance classifier
python train_article_group.py
--> Review article categorizations in categorized_articles.csv
    Mark articles to:
    - Keep (K)
    - Drop (D)

# Process final decisions
python process_user_categorizations.py
```

## Requirements

- pandas
- numpy 
- scikit-learn
- nltk
- networkx
- datasketch
- beautifulsoup4
- python-Levenshtein
- requests

## Citation

```
@software{abstract_scholar,
  author = {Pichardo, Felix},
  title = {Abstract Scholar},
  year = {2024},
  url = {https://github.com/fpichardo-umn/AbstractScholar}
}
```
