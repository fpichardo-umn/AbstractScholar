# -*- coding: utf-8 -*-

"""
STEP 01: Preprocess Abstracts

This script is designed to preprocess a collection of scientific article abstracts to prepare them for use in topic modeling. Specifically, it applies several text preprocessing techniques, including text cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization, before performing singular value decomposition (SVD) to extract the most important concepts or topics from the data.

Configuration parameters (defined in the user configuration file):
- ngram_range: Tuple specifying the minimum and maximum n-gram size to include in the analysis (default: (1, 4)).
- max_doc_freq: Float specifying the maximum document frequency for a term to be included in the analysis (default: 0.5).
- max_svd_components: Integer specifying the maximum number of components (topics) to extract from the data (default: 250).
- custom_stopwords: List of words/terms to ignore during the analysis (default includes 'al' and 'et').
- preprocess_pickle: Path to save the preprocessed data (default: './data/text_analysis/preprocessed_abstracts.pickle').

The script will save the resulting data (including the SVD results, the transformed document-term matrix, and the TF-IDF terms) to a file called 'preprocessed_abstracts.pickle'.

Created on Fri Feb 23 02:16:48 2018

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
# This imports utility and basic modules functions from the 'scripts' directory
scripts_dir = op.join(os.getcwd(), 'scripts')
sys.path.append(scripts_dir)
from scripts.gen_utils_text import *

# Script specific imports
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def remove_punc(text):
    """Remove punctuation from text data"""

    for punc in string.punctuation.replace('-', '').replace('.', ''):
        text = text.replace(punc, ' ')

    text = re.sub(r'[^\x00-\x7f]',r'', text)

    return re.sub(r'([^\d])\.([^\d])', r'\1\2', text)


def clean_nums(text):
    """Clean the numbers from the given text"""
    return re.sub(r'(^|\s)\d+(\s|$)', r' ', text)


def clean_text(txt):
    """Clean text data by removing non-printable characters and converting to ASCII"""
    
    non_printable_chars = ['\n', '\t', '‐']
    for to_remove in non_printable_chars:
        txt = txt.replace(to_remove, ' ')

    txt = ' '.join(txt.split())
    txt = str.encode(txt, 'ascii', 'ignore')

    return str(txt.strip())[2:-2]


def clean_column(column):
    """
    Clean the given column by performing the following steps:
    1. Convert all values to lowercase.
    2. Remove punctuation from each value.
    3. Clean up any numbers in each value.
    4. Split each value by semicolon (;).
    5. Remove duplicate words and leading/trailing whitespaces from each value.
    6. Join the cleaned words back into a single string.
    
    Parameters:
    - column (pandas.Series): The column to be cleaned.
    
    Returns:
    - pandas.Series: The cleaned column.
    """
    
    cleaned = column.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).str.split(';')
    cleaned = cleaned.apply(lambda x: list(dict.fromkeys([item.strip() for item in x[0].split()])))
    return cleaned.apply(lambda x: ' '.join(x))


def preprocess_data(data):
    """
    Preprocesses the data by cleaning and transforming the abstracts.

    Parameters:
    - data (pandas.DataFrame): The input data containing the abstracts.

    Returns:
    - clean_docs (list): A list of preprocessed documents.
    """
    
    data['abstract'] = data['abstract'].fillna('')
    data = data[data.abstract.apply(len) > 50]
    data = data.drop_duplicates('title').reset_index(drop=True)
    pd_idx = data.index.tolist()

    keywords = clean_column(data.keywords)
    authors = clean_column(data.authors.apply(clean_text))
    journals = clean_column(data.journal)

    docs = data.loc[pd_idx, 'abstract'] + ' ' + data.loc[pd_idx, 'title'] + ' ' + keywords + ' ' + authors + ' ' + journals
    clean_docs = [remove_punc(doc.lower()) for doc in docs]
    clean_docs = [clean_nums(doc) for doc in clean_docs]
    clean_docs = [word_tokenize(doc) for doc in clean_docs]
    return clean_docs


def text_preprocess(clean_docs, custom_stopwords):
    """
    Preprocesses the given documents by performing lemmatization and generating stopwords.

    Parameters:
    clean_docs (list): A list of documents to be preprocessed.
    custom_stopwords (list): A list of custom stopwords to be added to the default set of stopwords.

    Returns:
    tuple: A tuple containing the lemmatized vector and the set of stopwords.
    """
    # Create a lemmatizer object
    lemmatizer = WordNetLemmatizer()
    
    ##STOPWORDS
    # Define a set of stopwords to remove from the documents
    stop_words = set(stopwords.words('english'))

    # Add custom stopwords
    stop_words.update(custom_stopwords)
    stop_words.update(set(stopwords.words('spanish')))

    ##LEMMATIZE
    # Lemmatize stopwords and add to stop_words
    lem_stop = list(map(lemmatizer.lemmatize, stop_words))
    stop_words.update(lem_stop)

    lemmatized_docs   = [list(map(lemmatizer.lemmatize, doc)) for doc in clean_docs]
    lemmatized_vector = ['' for n in range(len(lemmatized_docs))]

    # Apply lemmatization to each word in the document and join back into a string
    for idx, lem_dos in enumerate(lemmatized_docs):
        for lem_word in lem_dos:
            if len(lem_word) > 1:
                lemmatized_vector[idx] += ' ' + lem_word
            
    lemmatized_vector = [clean_nums(doc) for doc in lemmatized_vector]
    
    return lemmatized_vector, stop_words


def tfidf_vectorize(lemmatized_vector, max_doc_freq, stop_words, ngram_range):
    """
    Vectorize the lemmatized documents using TF-IDF (Term Frequency-Inverse Document Frequency).

    Parameters:
    lemmatized_vector : list
        A list of lemmatized documents.
    max_doc_freq : float
        The maximum document frequency. Terms with a document frequency higher than this value will be ignored.
    stop_words : set
        A set of stop words to be removed from the documents.
    ngram_range : tuple
        The range of n-grams to be considered. For example, (1, 2) means both unigrams and bigrams will be considered.

    Returns:
    vectorizer_tfidf : TfidfVectorizer
        The fitted TF-IDF vectorizer.
    doc_term_mat : scipy.sparse.csr_matrix
        The document-term matrix representing the TF-IDF values.
    terms : list
        The terms in the vocabulary.
    """
    # The optimal min_doc_freq value is typically around 1-2% of the total number of documents.
    min_doc_freq = int(np.mean([len(lemmatized_vector) * 0.01, len(lemmatized_vector) * 0.02]))
    
    vectorizer_tfidf = TfidfVectorizer(
        max_df=max_doc_freq,
        min_df=min_doc_freq,
        stop_words=list(stop_words),
        use_idf=True,
        ngram_range=ngram_range,
        token_pattern=r'(?u)\b\w*[a-zA-Z]+\w*\b'
    )
    
    # Fit the vectorizer to the lemmatized documents and transform the documents into a document-term matrix
    doc_term_mat = vectorizer_tfidf.fit_transform(lemmatized_vector)
    
    # Get the terms in the vocabulary
    terms = vectorizer_tfidf.get_feature_names_out()
    
    return vectorizer_tfidf, doc_term_mat, terms


def make_svd(doc_term_mat, n_components, remove_0=False):
    """
    Perform truncated SVD on a document-term matrix.

    Parameters:
    - doc_term_mat: The document-term matrix to perform SVD on.
    - n_components: The number of components to keep.
    - remove_0: A boolean indicating whether to remove the first component.

    Returns:
    - latent_sa: The fitted TruncatedSVD object.
    - doc_term_mat_latent_sa: The transformed document-term matrix after SVD.
    """
    latent_sa = TruncatedSVD(n_components, algorithm='arpack')
    latent_sa = latent_sa.fit(doc_term_mat)
    if remove_0:
        latent_sa.components_[0] = 0
    doc_term_mat_latent_sa = latent_sa.transform(doc_term_mat)
    doc_term_mat_latent_sa = Normalizer(copy=False).fit_transform(doc_term_mat_latent_sa)

    return latent_sa, doc_term_mat_latent_sa


def perform_svd(doc_term_mat, max_svd_components, plot_svd=True):
    """
    Perform Singular Value Decomposition (SVD) on the given document-term matrix
    and select the optimal number of components based on the explained variance ratio.
    
    Parameters:
        doc_term_mat (numpy.ndarray): The document-term matrix to perform SVD on.
        max_svd_components (int): The maximum number of SVD components to consider.
        plot_svd (bool, optional): Whether to plot the explained variance ratio. Defaults to True.
    
    Returns:
        tuple: A tuple containing the latent semantic analysis (LSA) and the transformed document-term matrix.
    """
    
    # Compute original SVD
    latent_sa, doc_term_mat_xfm = make_svd(doc_term_mat, min(doc_term_mat.shape) - 1)
    
    if plot_svd:
        plt.plot(latent_sa.explained_variance_ratio_[:max_svd_components])
    
    # The threshold is set to 0.001, meaning that components with an explained variance ratio below 0.001 will be discarded
    # This helps to select the most important components that capture the majority of the information in the data
    components_above_threshold = np.where(latent_sa.explained_variance_ratio_ >= 0.001)[0]
    
    # Check if any components meet the criterion else consider all components
    opt_component_cnt = components_above_threshold[-1] + 1 if components_above_threshold.size > 0 else len(latent_sa.explained_variance_ratio_)
    print(f"Optimal number of components: {opt_component_cnt}")
    
    # Perform SVD with the optimal number of components
    latent_sa, doc_term_mat_xfm = make_svd(doc_term_mat, opt_component_cnt)
    
    return latent_sa, doc_term_mat_xfm



####
##    START
####

# Load preprocessed data
config = load_user_config()
data = load_data(config)

# Get the configuration parameters
ngram_range = tuple(map(int, config.get('ngram_range', '1, 4').split(',')))
max_doc_freq = float(config.get('max_doc_freq', 0.5))
max_svd_components = int(config.get('max_svd_components', 250))
custom_stopwords = config.get('custom_stopwords', 'al, et, contents'.split(',')) #['al', 'et', 'contents', 'pubmed', 'pnas', 'published', 'article', 'present', 'abstract', 'sp', 'american', 'psychological', 'association', 'associatio', 'ei', 'pg', 'user', 'ie', 'apa', 'rights', 'reserved', 'copyright', 'c', 'l', 'j']
preprocess_pickle_filename = normalize_path(config.get('preprocess_pickle', './data/text_analysis/large_files/preprocessed_abstracts.pickle'))

# Preprocess the data by cleaning and transforming the abstracts
clean_docs = preprocess_data(data)

# Perform text preprocessing by lemmatizing the documents and generating stopwords
lemmatized_vector, stop_words = text_preprocess(clean_docs, config)

# Vectorize the preprocessed data using TF-IDF
vectorizer_tfidf, doc_term_mat, terms = tfidf_vectorize(lemmatized_vector, config)

# Perform Singular Value Decomposition (SVD) on the document-term matrix
latent_sa, doc_term_mat_xfm = perform_svd(doc_term_mat, config)

###
#   Check and clean terms
##
# Get the top terms associated with each concept in the SVD
top_terms_per_concept = []
for ii in range(latent_sa.components_.shape[0]):
    top_terms_per_concept.extend(get_terms_for_concept(terms, latent_sa.components_, ii)[:10])
top_terms_per_concept = list(set(top_terms_per_concept))

# Serialize the preprocessed data
save_to_pickle([latent_sa, doc_term_mat_xfm, terms], preprocess_pickle_filename)