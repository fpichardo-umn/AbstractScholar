# -*- coding: utf-8 -*-

"""
STEP 01: Preprocess Abstracts

This script is designed to preprocess a collection of scientific article abstracts to prepare them for use in topic modeling. Specifically, it applies several text preprocessing techniques, including keyword filtering, cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization, before performing singular value decomposition (SVD) to extract the most important concepts or topics from the data.

To use this script, you'll need to modify some of the parameters at the top of the script to suit your specific needs. These include:

In the user configuration file, modify the following parameters:
ngram_range: a tuple specifying the minimum and maximum n-gram size to include in the analysis. The default is (1, 4), which means that unigrams, bigrams, trigrams, and quadgrams will be included.
max_doc_freq: a float specifying the maximum document frequency for a term to be included in the analysis. The default is 0.5, which means that terms that appear in more than half of the documents will be excluded.
max_svd_components: an integer specifying the maximum number of components (i.e., topics) to extract from the data. The default is 250.
custom_stopwords: a list of words/terms to ignore during the analysis. The default includes 'al' and 'et'

The script will save the resulting data (including the SVD results, the TF-IDF terms, and the original abstracts) to a file called 'preprocessed_abstracts.pickle'.

Created on Fri Feb 23 02:16:48 2018

@author: Felix Pichardo
"""

import sys
import os
import os.path as op

# Add the 'scripts' subdirectory to the Python path
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

    for to_remove in non_printable_chars:
        txt = txt.replace(to_remove, ' ')

    txt = ' '.join(txt.split())
    txt = str.encode(txt, 'ascii', 'ignore')

    return str(txt.strip())[2:-2]


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



####
##    START
####

# Load preprocessed data
config = load_user_config()

# Load preprocessed datas
data = load_data(config)

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Define a list of non-printable characters to clean from text
non_printable_chars = ['\n', '\t', '‐']

# Remove any rows with a short 'abstract' (less than 50 characters)
data['abstract'] = data['abstract'].fillna('')
data = data[data.abstract.apply(len) > 50]

# Remove any rows with duplicate 'Title' values
data   = data.drop_duplicates('Title').reset_index(drop = True)
pd_idx = data.index.tolist()

# Clean up and remove dupe keywords
keywords = data.keywords.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).str.split(';')
keywords = keywords.apply(lambda x: list(dict.fromkeys([keyword.strip() for keyword in x[0].split()])))
keywords = keywords.apply(lambda x: ' '.join(x))

# Clean up and remove dupe authors
authors = data.authors.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).apply(clean_text).str.split(';')
authors = authors.apply(lambda x: list(dict.fromkeys([author.strip() for author in x[0].split()])))
authors = authors.apply(lambda x: ' '.join(x))

# Clean up and remove dupe journals
journals = data.journal.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).str.split(';')
journals = journals.apply(lambda x: list(dict.fromkeys([journal.strip() for journal in x[0].split()])))
journals = journals.apply(lambda x: ' '.join(x))

# Combine the abstract, title, keywords, authors, and journals into a single string for each document
docs = data.loc[pd_idx, 'abstract'] + ' ' + data.loc[pd_idx, 'Title'] + ' ' + keywords + ' ' + authors + ' ' + journals

# Remove punctiation and get rid of non-printable characters
clean_docs = [remove_punc(doc.lower()) for doc in docs]

# Tokenize the documents and remove any numbers
clean_docs = [clean_nums(doc) for doc in clean_docs]
clean_docs = [word_tokenize(doc) for doc in clean_docs]

##STOPWORDS
# Define a set of stopwords to remove from the documents
stop_words = set(stopwords.words('english'))

# Add custom stopwords
custom_stopwords = config.get('custom_stopwords', 'al, et, contents'.split(','))
#['al', 'et', 'contents', 'pubmed', 'pnas', 'published', 'article', 'present', 'abstract', 'sp', 'american', 'psychological', 'association', 'associatio', 'ei', 'pg', 'user', 'ie', 'apa', 'rights', 'reserved', 'copyright', 'c', 'l', 'j']
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

##TFIDF
## Recompute min_doc_freq:
#The optimal min_doc_freq might be slightly dataset-specific but generally, a value
# that reduces the vocabulary to terms that appear in at least 1-2% of the 
#documents can significantly reduce noise without losing valuable information.
min_doc_freq = int(np.mean([len(lemmatized_vector)*0.01, len(lemmatized_vector)*0.02]))

vectorizer_tfidf = TfidfVectorizer(max_df=float(config.get('max_doc_freq', 0.5)),
                                   min_df=min_doc_freq,
                                   stop_words=list(stop_words), use_idf = True,
                                   ngram_range=tuple([int(num) for num in config.get('ngram_range', '1, 4').split(',')]),
                                   token_pattern = r'(?u)\b\w*[a-zA-Z]+\w*\b')

# Fit the vectorizer to the lemmatized documents and transform the documents into a document-term matrix
doc_term_mat = vectorizer_tfidf.fit_transform(lemmatized_vector)

# Get the terms in the vocabulary
terms = vectorizer_tfidf.get_feature_names_out()

#SVD
# Compute the SVD
latent_sa, doc_term_mat_xfm = make_svd(doc_term_mat, min(doc_term_mat.shape)-1)
plt.plot(latent_sa.explained_variance_ratio_[:int(config.get('max_svd_components', 250))])

# Compute number of components to keep
# Filter to find the last component that is above the threshold
components_above_threshold = np.where(latent_sa.explained_variance_ratio_ >= 0.001)[0]

# Check if any components meet the criterion
if components_above_threshold.size > 0:
    opt_component_cnt = components_above_threshold[-1] + 1  # +1 because index is 0-based
else:
    # If no components meet the criterion, consider all components
    opt_component_cnt = len(latent_sa.explained_variance_ratio_)

print(f"Optimal number of components: {opt_component_cnt}")

latent_sa, doc_term_mat_xfm = make_svd(doc_term_mat, opt_component_cnt)


###
#   Check and clean terms
##
# Get the top terms associated with each concept in the SVD
top_terms_per_concept = []
for ii in range(opt_component_cnt):
    top_terms_per_concept.extend(get_terms_for_concept(terms, latent_sa.components_, ii)[:10])

# Remove duplicated terms to check
top_terms_per_concept = list(set(top_terms_per_concept))

# Serialize the preprocessed data
save_to_pickle([latent_sa, doc_term_mat_xfm, terms], config.get('preprocess_pickle', './data/preprocessed_abstracts.pickle'))