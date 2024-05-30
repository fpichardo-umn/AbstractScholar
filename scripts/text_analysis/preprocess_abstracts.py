# -*- coding: utf-8 -*-
"""
STEP 01: Preprocess Abstracts

This script is designed to preprocess a collection of scientific article abstracts to prepare them for use in topic modeling. Specifically, it applies several text preprocessing techniques, including keyword filtering, cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization, before performing singular value decomposition (SVD) to extract the most important concepts or topics from the data.

To use this script, you'll need to modify some of the parameters at the top of the script to suit your specific needs. These include:

ngram_range: a tuple specifying the minimum and maximum n-gram size to include in the analysis. The default is (1, 4), which means that unigrams, bigrams, trigrams, and quadgrams will be included.
min_df: an integer specifying the minimum document frequency for a term to be included in the analysis. The default is 2, which means that terms must appear in at least two documents to be included.
max_df: a float specifying the maximum document frequency for a term to be included in the analysis. The default is 0.5, which means that terms that appear in more than half of the documents will be excluded.
max_components: an integer specifying the maximum number of components (i.e., topics) to extract from the data. The default is 250.
imp_keywords: a list of keywords that will be used to filter the abstracts to include only those that are most relevant. The default is an empty list, which means that no keyword filtering will be applied.
to_clean_strs: a list of strings that will be removed from the abstracts before further processing. The default includes newline characters, tab characters, and a Unicode dash character.
Once you have modified these parameters to your liking, you can run the script. It will read in a file called 'DATA.txt', which should contain a tab-separated file with columns for 'Title' and 'Excerpt' (which should contain the abstracts). The script will clean and preprocess the abstracts, perform SVD to extract the most important concepts, and save the resulting data (including the SVD results, the TF-IDF terms, and the original abstracts) to a file called 'preprocessed_abstracts.pickle'.

You can then use this preprocessed data to perform topic modeling using a method of your choice (e.g., latent Dirichlet allocation, non-negative matrix factorization, etc.).

Created on Fri Feb 23 02:16:48 2018

@author: Felix Pichardo
"""

import pickle
import re
import string
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def remove_punc(text):
    """
    Remove punctuation from text data.
    """

    for punc in string.punctuation.replace('-', '').replace('.', ''):
        text = text.replace(punc, ' ')

    text = re.sub(r'[^\x00-\x7f]',r'', text)

    return re.sub(r'([^\d])\.([^\d])', r'\1\2', text)


def clean_nums(text):

    return re.sub(r'(^|\s)\d+(\s|$)', r' ', text)


def display_topics_tab(model, feature_names, no_top_words):

    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print( " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        input()


def get_article_top_concepts(docs, concepts_transformed_data):

    art_top_concepts = {art:[] for art in range(len(docs))}
    top_num = 5
    for i in range(len(concepts_transformed_data)):
       top_topics = np.argsort(concepts_transformed_data[i,:])[::-1][:top_num]
       top_topics_str = ' '.join(str(t) for t in top_topics)
       art_top_concepts[i] = [int(top) for top in top_topics_str.split()]

    return art_top_concepts


def get_terms_for_concept(terms, concept_matrix, concept_idx, top_n = 10):
    """
    Get top n terms for a given concept in a concept matrix.
    """

    concept_scores = concept_matrix[concept_idx]
    top_indices = np.argsort(concept_scores)[::-1][:top_n]
    top_terms = [terms[i] for i in top_indices]

    return top_terms


def get_top_words_art(art_idx_list, terms, components, concepts_per_art = 3, terms_per_concept = 10):

    if not 'article_top_concepts' in dir():
        article_top_concepts = get_article_top_concepts(docs, dtm_lsa)

    concepts = []
    for art in art_idx_list:
        concepts.extend(article_top_concepts[art][:concepts_per_art])

    concepts = list(set(concepts))

    top_terms = []
    for concept in concepts:
        top_terms.append(get_terms_for_concept(terms, components, concept, terms_per_concept))

    return top_terms


def make_svd(dtm, n_components, remove_0 = False):
    """
    Perform truncated SVD on a document-term matrix.
    """

    lsa = TruncatedSVD(n_components, algorithm = 'arpack')
    lsa = lsa.fit(dtm)
    if remove_0:
        lsa.components_[0] = 0
    dtm_lsa = lsa.transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    return lsa, dtm_lsa


def clean_text(txt):
    """
    Clean and preprocess text data by removing non-printable characters,
    and replacing specific characters with more common ones.
    """

    for to_remove in to_clean_strs:
        txt = txt.replace(to_remove, ' ')

    txt = ' '.join(txt.split())
    txt = str.encode(txt, 'ascii', 'ignore')

    return str(txt.strip())[2:-2]


def keyword_filter(abstracts, keywords):
    """
    Filter abstracts based on a list of keywords and return idx to keep
    """

    keep_bool = [False for abstract in abstracts]
    for keyword in keywords:
        for idx, abstract in enumerate(abstracts):
            keep_bool[idx] |= keyword in abstract.lower()

    keep_idx = [idx for idx, val in enumerate(keep_bool) if val]

    return keep_idx


####
##    START
####

# Set the range for the n-grams
ngram_range = (1, 4)

# Set the minimum document frequency for a term to be included in the corpus
min_df = 4

# Set the maximum document frequency for a term to be included in the corpus
max_df = 0.5

# Set the maximum number of components to use in SVD
max_components = 250

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Define a list of non-printable characters to clean from text
to_clean_strs = ['\n', '\t', '‐']

# Define a regular expression pattern for matching tokens of at least 3 characters
token_3_chars_re = r'[a-zA-Z0-9]{2,}'

# Read in the data file as a pandas DataFrame
pulled_data = 'pulled_abs.txt'
data = pd.read_csv(op.join('data', pulled_data), sep = '\t', encoding='iso-8859-1', index_col = None)

# Fill any NaN values in the 'Excerpt' column with an empty string
# data.Excerpt.fillna('', inplace=True)

# Identify any rows with a short 'Excerpt' and copy the 'Excerpt' to the 'abstract' column
# to_add_excerpt = data.loc[data.abstract.apply(str).apply(len) < 50].index
# days_ago_re = r'\d+ days ago -? +'
# clean_excerpt = lambda exceprt: re.sub(days_ago_re, '', clean_text(exceprt))
# data.loc[to_add_excerpt.values, 'abstract'] = data.loc[to_add_excerpt.values, 'Excerpt'].apply(clean_excerpt)

# Remove any rows with a short 'abstract' (less than 50 characters)
data['abstract'] = data['abstract'].fillna('')
data = data[data.abstract.apply(len) > 50]

# Remove any rows with duplicate 'Title' values
data = data.drop_duplicates('Title').reset_index(drop = True)

# Create a list of the indices and abstracts to keep
pd_idx, docs = [], []
for idx, abstract in enumerate(data.abstract):
    if 'no abstract in url' not in abstract.lower():
        pd_idx.append(idx)
        docs.append(abstract)


# Clean up and add keywords
keywords = data.keywords.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).str.split(';')

# Split each string within the list into individual keywords and remove duplicates
keywords = keywords.apply(lambda x: list(dict.fromkeys([keyword.strip() for keyword in x[0].split()])))
keywords = keywords.apply(lambda x: ' '.join(x))

# Clean up and add authors
authors = data.authors.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).apply(clean_text).str.split(';')

# Split each string within the list into individual keywords and remove duplicates
authors = authors.apply(lambda x: list(dict.fromkeys([author.strip() for author in x[0].split()])))
authors = authors.apply(lambda x: ' '.join(x))

# Clean up and add authors
journals = data.journal.fillna('').apply(lambda x: x.lower()).apply(remove_punc).apply(clean_nums).str.split(';')

# Split each string within the list into individual keywords and remove duplicates
journals = journals.apply(lambda x: list(dict.fromkeys([journal.strip() for journal in x[0].split()])))
journals = journals.apply(lambda x: ' '.join(x))


# Combine the abstract and title into a single string for each document
abstracts = data.loc[pd_idx, 'abstract'] + ' ' + data.loc[pd_idx, 'Title'] + ' ' + keywords + ' ' + authors + ' ' + journals

##FILTER
# Filter the documents based on a list of important keywords
# imp_keywords = [''] #MODIFY THIS
# keyword_filter_idx = keyword_filter(abstracts, imp_keywords)
# docs = data.loc[keyword_filter_idx, 'abstract'].values

docs = abstracts

##IF NO KEYWORD FILTER
#docs = data.abstract.values


# Create a list of the cleaned 'Title' values
doc_short = [clean_text(data.Title.iloc[idx]) for idx in pd_idx]

# Remove punctiation and get rid of non-printable characters
clean_docs = [remove_punc(doc.lower()) for doc in docs]
doc_short  = [remove_punc(doc) for doc in doc_short]

# Tokenize the documents and remove any numbers
clean_docs = [clean_nums(doc) for doc in clean_docs]
clean_docs = [word_tokenize(doc) for doc in clean_docs]

##STOPWORDS
# Define a set of stopwords to remove from the documents
stop_words = set(stopwords.words('english'))

# Add custom stopwords
custom_stopwords = ['al', 'et', 'contents', 'pubmed', 'pnas', 'published', 'article', 'present', 'abstract', 'sp', 'american', 'psychological', 'association', 'associatio', 'ei', 'pg', 'user', 'ie', 'apa', 'rights', 'reserved', 'copyright', 'c', 'l', 'j']
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
## Recompute min_df:
#The optimal min_df might be slightly dataset-specific but generally, a value
# that reduces the vocabulary to terms that appear in at least 1-2% of the 
#documents can significantly reduce noise without losing valuable information.
min_df = int(np.mean([len(lemmatized_vector)*0.01, len(lemmatized_vector)*0.02]))
vectorizer_tfidf = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                   stop_words=list(stop_words), use_idf = True,
                                   ngram_range=ngram_range,
                                   token_pattern = r'(?u)\b\w*[a-zA-Z]+\w*\b')

# Fit the vectorizer to the lemmatized documents and transform the documents into a document-term matrix
dtm = vectorizer_tfidf.fit_transform(lemmatized_vector)

# Get the terms in the vocabulary
terms = vectorizer_tfidf.get_feature_names_out()

#SVD
# Define the minimum variance ratio to retain in the SVD
var_ratio_min = len(terms) * (.13/16325) #Based on 26 in original data
while var_ratio_min < .1:
    var_ratio_min *= 2

# Compute the SVD with the given minimum variance ratio
lsa, dtm_lsa = make_svd(dtm, min(dtm.shape)-1)
plt.plot(lsa.explained_variance_ratio_[:max_components])
lsa_df = pd.DataFrame(lsa.explained_variance_ratio_).cumsum()

# # Choose the number of components to retain in the SVD based on the minimum variance ratio
# n_trunc = 0
# while n_trunc < 5 or np.isnan(n_trunc):
#     n_trunc = lsa_df.loc[lsa_df[0] <= var_ratio_min].index.max()
#     if n_trunc < 5 or np.isnan(n_trunc):
#         var_ratio_min *= 2
#del lsa_df

# Assuming lsa.explained_variance_ratio_ contains the explained variance ratio for each component
cumulative_explained_variance = lsa.explained_variance_ratio_.cumsum()

# Initialize n_trunc to 0
n_trunc = 0

# Iterate through cumulative explained variance
for idx, variance in enumerate(cumulative_explained_variance):
    # Check if the incremental contribution from the last component is below 0.1%
    if idx > 0 and (cumulative_explained_variance[idx] - cumulative_explained_variance[idx-1]) < 0.001:
        # Set n_trunc to the current index (components are 0-indexed, so add 1 for the actual count)
        n_trunc = idx
        break

# If no components meet the criterion, consider all components
if n_trunc == 0:
    n_trunc = len(cumulative_explained_variance)

print(f"Optimal number of components: {n_trunc}")

lsa, dtm_lsa = make_svd(dtm, n_trunc)


###
#   Check and clean terms
##
# Get the top terms associated with each concept in the SVD
check_terms = []
for ii in range(n_trunc):
    check_terms.extend(get_terms_for_concept(terms, lsa.components_, ii)[:10])

# Remove duplicated terms to check
check_terms = list(set(check_terms))

#term_to_check = {}
#for term in check_terms:
#    print(term)
#    inp = input('keep? ')
#    term_to_check[term] = inp
#    print()

# Serialize the preprocessed data
preprocess_pickle = op.join('data', 'preprocessed_abstracts.pickle')
with open(preprocess_pickle, 'wb') as p:
    pickle.dump([lsa, dtm_lsa, terms], p)
