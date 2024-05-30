# -*- coding: utf-8 -*-
"""
STEP 04: Train Article Group Classifier

Created on Sat Apr 14 15:47:29 2018

@author: Felix Pichardo
"""

import pickle
import pandas as pd
import numpy as np
import os.path as op
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans


def eval_cluster(docs, dtm_lsa, cluster_num):
    
    cluster_idx = docs.loc[docs.group == cluster_num].index
    cluster_data = dtm_lsa[cluster_idx, :]
    
    print('Num articles:{}\n'.format(len(cluster_idx)))
    
    cos_dist = cosine_distances(cluster_data)
    
    print('Most dissimilar value:{}\n'.format(cos_dist.max()))
    cluster_dism_idx = [list(val) for val in np.where(cos_dist == cos_dist.max())]
    
    uniq_dism_idx = []
    for idx, vals in enumerate(cluster_dism_idx):
        if vals not in uniq_dism_idx and vals[::-1] not in uniq_dism_idx:
            uniq_dism_idx.append(vals)
    
    cluster_dism_titles = [docs.Title[cluster_idx[[idx]]].values
                           for idx in uniq_dism_idx]
    
    print(cluster_dism_titles)
    
    dism_sum = abs(cos_dist.sum(axis = 0))
    
    least_dism = np.where(dism_sum == dism_sum.min())
    least_dism_title = docs.Title[cluster_idx[least_dism]].values[0]
    print('Most similar:', least_dism_title)
    
    most_dism = np.where(dism_sum == dism_sum.max())
    most_dism_title = docs.Title[cluster_idx[most_dism]].values[0]
    print('Most disimilar:{}\n'.format(most_dism_title))
    
    cluster_cmp_avg =  cluster_data.mean(axis = 0) * avg_centroid
    cluster_gen = np.dot(cluster_data, cluster_cmp_avg)
    gen_idx = np.where(cluster_gen == cluster_gen.max())[0][0]
    most_gen = docs.Title[cluster_idx[gen_idx]]
    print('Most general:{}\n'.format(most_gen))
    
    to_read = [least_dism_title, most_dism_title]
    to_check = cluster_dism_titles[0].tolist() + [most_gen]
    for art in to_check:
        print(art)
        if art not in to_read:
            to_read.append(art)
    
    return to_read


def make_svd(dtm, n, remove_0 = True):
    
    lsa = TruncatedSVD(n, algorithm = 'arpack')
    lsa = lsa.fit(dtm)
    if remove_0:
        lsa.components_[0] = 0
    dtm_lsa = lsa.transform(dtm)
    
    return lsa, dtm_lsa


def get_terms_for_concept(terms, components, concept_idx, num_terms = 20):

    comp = components[concept_idx]
    terms_in_comp = zip(terms, comp)
    sorted_terms = sorted(terms_in_comp, key = lambda x: x[1], reverse = True) [:num_terms]
    sorted_terms = [term[0] for term in sorted_terms]
    
    return sorted_terms


def get_cluster_top_terms(cluster_id, cluster_centroids, lsa, terms, num_terms = 20):
    
    #Get centroid
    cluster_centroid = cluster_centroids[cluster_id]
    
    #Get term weights for cluster
    cluster_term_weights = np.dot(cluster_centroid, lsa.components_)
    cluster_term_weights = np.abs(cluster_term_weights) #need absolute values for the weights because of the sign indeterminacy in LSA
    sorted_terms_idx = np.argsort(cluster_term_weights)[::-1][:num_terms]
    top_terms = [terms[idx] for idx in sorted_terms_idx]
    
    cnt_sub_in_terms = lambda x: sum([1 for term in top_terms if x in term])
    cleaned_top_terms = [term for term in top_terms if cnt_sub_in_terms(term) == 1]
    
    return cleaned_top_terms


def normalize_values(coherences):
    # Get maximum coherence value
    max_value = max(coherences.values())
    
    # Normalize values
    normalized_coherences = {key: value / max_value for key, value in coherences.items()}
    
    return normalized_coherences





###
#   Start 
###

data_for_edist = op.join('data', 'clustered_data.txt')
clustered_data = op.join('data', 'clustered_sample.txt')
data = pd.read_csv(data_for_edist, sep = '\t', encoding='iso-8859-1', index_col = None)
sample_df = pd.read_csv(clustered_data, sep = '\t', encoding='iso-8859-1', index_col = None)

preprocess_pickle = op.join('data', 'preprocessed_abstracts.pickle')
with open(preprocess_pickle, 'rb') as p:
    lsa, dtm_lsa, terms = pickle.load(p)


cluster_pickle = op.join('data', 'cluster_abstracts.pickle')
with open(cluster_pickle, 'rb') as p:
    group_centroids, merged_clusters_info, updated_clusters_dtm_rows, \
        updated_cluster_names, updated_cluster_titles, sorted_all_coherences = \
            pickle.load(p)


# ML
# Define relevant and irrelevant clusters
relevant_clusters = [5, 13, 'merged_33_8']
irrelevant_clusters = [14, 3, 36]

# Initialize labels
data['label'] = None

# Label articles based on cluster membership
for cluster in relevant_clusters:
    data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 1
for cluster in irrelevant_clusters:
    data.loc[data.index.isin(merged_clusters_info[cluster]['indices']), 'label'] = 0

# Assuming 'dtm_lsa' is your document-term matrix after applying LSA/SVD
# 'group_centroids' is a dictionary with cluster IDs as keys and centroids as values
normalized_dict = normalize_values(dict(sorted_all_coherences))
for cluster_id, centroid in group_centroids.items():
    indices = merged_clusters_info[cluster_id]['indices']
    data.loc[indices, 'distance_to_centroid'] = cosine_distances(dtm_lsa[indices], centroid.reshape(1, -1)).flatten()
    data.loc[indices, 'coherence_weight'] = normalized_dict[cluster_id]  # Assuming coherence scores are normalized

# Assuming 'dtm_lsa' is your document-term matrix after applying SVD/LSA
labeled_data = data.dropna(subset=['label'])
# Extract SVD components for labeled data
labeled_features = dtm_lsa[labeled_data.index]

# Combine with other features
features = np.hstack([labeled_features, labeled_data[['distance_to_centroid', 'coherence_weight']].values])


# Labels based on cluster relevance: 1 for relevant, 0 for irrelevant, and possibly NaN for undetermined
labels = labeled_data['label']  # Ensure this column is already populated based on cluster relevance


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

enet = ElasticNetCV(cv=5)  # Set cv=5 for 5-fold cross-validation
enet.fit(X_train, y_train)

# Predict relevance on the test set
predictions = enet.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions_binary)
precision = precision_score(y_test, predictions_binary)
recall = recall_score(y_test, predictions_binary)
f1 = f1_score(y_test, predictions_binary)

# Print out the performance
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Select unlabeled data
unlabeled_data = data.copy()

unlabeled_data.loc[:, 'nearest_centroid'] = None
unlabeled_data.loc[:, 'distance_to_nearest_centroid'] = np.inf

for cluster_id, centroid in group_centroids.items():
    distances = cosine_distances(dtm_lsa[unlabeled_data.index], centroid.reshape(1, -1)).flatten()
    for idx, distance in zip(unlabeled_data.index, distances):
        if distance < unlabeled_data.at[idx, 'distance_to_nearest_centroid']:
            unlabeled_data.at[idx, 'nearest_centroid'] = cluster_id
            unlabeled_data.at[idx, 'distance_to_nearest_centroid'] = distance

# 'normalized_dict' contains normalized coherence scores for all clusters
for idx in unlabeled_data.index:
    cluster_id = unlabeled_data.at[idx, 'nearest_centroid']
    unlabeled_data.at[idx, 'coherence_weight'] = normalized_dict.get(cluster_id, 0)  # Default to 0 if cluster_id not found


# Extract features for unlabeled data
unlabeled_features = np.hstack([
    dtm_lsa[unlabeled_data.index],
    unlabeled_data[['distance_to_nearest_centroid', 'coherence_weight']].values
])


# Predict relevance for unlabeled articles
unlabeled_predictions = enet.predict(unlabeled_features)

# Update your DataFrame with the predictions
unlabeled_data.loc[unlabeled_data.index, 'predicted_1_prob'] = 1 / (1 + np.exp(-unlabeled_predictions))
unlabeled_data.loc[unlabeled_data.index, 'predicted_label'] = (unlabeled_predictions > 0.5).astype(int)

# Calculate the final score considering distance, coherence, and ML probability
mean_std_normalize = lambda x: (x - x.mean()) / x.std()
min_max_normalize = lambda x: (x - x.min()) / (x.max() - x.min())

irrelevant_centroids = {cid: group_centroids[cid] for cid in relevant_clusters}

# Initialize columns for the nearest irrelevant centroid
unlabeled_data['sim_to_nearest_irrelevant_centroid'] = -np.inf
unlabeled_data['nearest_irrelevant_centroid_coherence'] = 0

# Calculate distances to irrelevant centroids and update the dataframe
for cluster_id, centroid in irrelevant_centroids.items():
    sims = cosine_similarity(dtm_lsa[unlabeled_data.index], centroid.reshape(1, -1)).flatten()
    for idx, sim in zip(unlabeled_data.index, sims):
        if sim > unlabeled_data.at[idx, 'sim_to_nearest_irrelevant_centroid']:
            unlabeled_data.at[idx, 'sim_to_nearest_irrelevant_centroid'] = distance
            unlabeled_data.at[idx, 'nearest_irrelevant_centroid_coherence'] = normalized_dict.get(cluster_id, 0)

# Adjust the final score based on the distance to the nearest irrelevant centroid and its coherence
unlabeled_data['final_score'] = min_max_normalize(unlabeled_data['predicted_1_prob'] - (unlabeled_data['sim_to_nearest_irrelevant_centroid'] * unlabeled_data['nearest_irrelevant_centroid_coherence'])**(1/8))

# Statistical Method
mean_score = unlabeled_data['final_score'].mean()
std_score = unlabeled_data['final_score'].std()
lower_threshold = mean_score - std_score
upper_threshold = mean_score + std_score

# Cluster-Based Method
kmeans = KMeans(n_clusters=3, random_state=42).fit(unlabeled_data[['final_score']])
clusters = kmeans.labels_
centroids = kmeans.cluster_centers_.flatten()
sorted_centroids = np.sort(centroids)


# Synthesizing Approaches
# Final thresholds are average of statistical and cluster-based methods
final_lower_threshold = (lower_threshold + sorted_centroids[0]) / 2
final_upper_threshold = (upper_threshold + sorted_centroids[-1]) / 2

# Assign categories based on final thresholds
unlabeled_data['category'] = np.where(unlabeled_data['final_score'] <= final_lower_threshold, 'Irrelevant',
                                      np.where(unlabeled_data['final_score'] >= final_upper_threshold, 'Relevant', 'Borderline'))

# Sample articles from each category
relevant_sample = unlabeled_data[unlabeled_data['category'] == 'Relevant'][['Title', 'abstract']].sample(5)
borderline_sample = unlabeled_data[unlabeled_data['category'] == 'Borderline'][['Title', 'abstract']].sample(5)
irrelevant_sample = unlabeled_data[unlabeled_data['category'] == 'Irrelevant'][['Title', 'abstract']].sample(5)

# Display samples
print("Relevant Articles Sample:\n", relevant_sample)
print("\nBorderline Articles Sample:\n", borderline_sample)
print("\nIrrelevant Articles Sample:\n", irrelevant_sample)






















# Gather article indices from relevant and irrelevant clusters
relevant_articles_idx = sum([merged_clusters_info[cluster]['indices'] for cluster in relevant_clusters], [])
irrelevant_articles_idx = sum([merged_clusters_info[cluster]['indices'] for cluster in irrelevant_clusters], [])

# Create labels: 1 for relevant, 0 for irrelevant
labels = [1] * len(relevant_articles_idx) + [0] * len(irrelevant_articles_idx)

# Combine indices and create the feature matrix
training_idx = relevant_articles_idx + irrelevant_articles_idx
X_train = dtm_lsa[training_idx, :]









# Calculate similarity of each article to each cluster's centroid
article_cluster_similarity = cosine_similarity(dtm_lsa, list(group_centroids.values()))

# Assuming merged_centroids is ordered as in article_cluster_similarity calculation
cluster_ids = list(group_centroids.keys())  # This should include both 'Y' and 'N' cluster IDs

# Map cluster IDs to their indices in the similarity array
cluster_id_to_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}


# Calculate average similarity to 'Y' and 'N' clusters
y_clusters = [2, 3, 16, '21-11', '5-15', '18-19']  # Example cluster IDs marked as 'Y'
n_clusters = [4, 10, 7]  # Example cluster IDs marked as 'N'

y_similarity = article_cluster_similarity[:, y_clusters].mean(axis=1)
n_similarity = article_cluster_similarity[:, n_clusters].mean(axis=1)













data['group'] = ''
avg_centroid = dtm_lsa.mean(axis=0)

##Train clf
X = dtm_lsa[sample_df.data_row, :]
y = sample_df.group
group_num = len(set(y))


from sklearn.model_selection import KFold
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
#
#param_grid = [
#        {'criterion': ['gini', 'entropy'],
#         'max_depth': range(2,20, 2),
#         'min_samples_split': range(2,5),
#         'min_samples_leaf': range(1,4),
#         'max_features': ['auto', 'log2', None],
#         'max_leaf_nodes': range(2,5),
#         'min_impurity_split': [1e-7, 1e-5, 1e-9, 1e-2, 1e-11]
#         }
#]
#
param_grid = [
        {'metric': ['euclidean', 'cosine', 'jaccard'],
         'shrink_threshold': [None, 0.1, 0.2, 0.5, .7]
        }
]
#
f1_scorer = make_scorer(f1_score, average = 'micro')
#DTC = DecisionTreeClassifier()
NC = NearestCentroid()
#ETC = ExtraTreesClassifier()
KF = KFold(n_splits=50, shuffle = True)
clf_search = GridSearchCV(NC, param_grid = param_grid, cv = KF, scoring = f1_scorer)
#
clf_search.fit(X, y = y)

###Best f1
#DTC = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
#            max_features='log2', max_leaf_nodes=4,
#            min_impurity_split=1e-05, min_samples_leaf=3,
#            min_samples_split=3, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best')
##Best NC
NC = NearestCentroid(metric='euclidean', shrink_threshold=None)

##Classify
mask = np.ones(data.shape[0], bool)
mask[sample_df.data_row] =  False
rows_to_classify = data.loc[mask].index

#Train classifier
clf = NC.fit(X, y = y)

print(get_cluster_top_terms(0, clf.centroids_, lsa, terms)[:20])

###
#  Classify
###

#Preprocess
docs= data.copy()
docs['group'] = 99
docs.loc[sample_df.data_row, 'group'] = y.values

dtm_to_class = dtm_lsa[rows_to_classify]

#Classify
classified = clf.predict(dtm_to_class)

#Update docs
docs.loc[rows_to_classify, 'group'] = classified


cluster_csv = 'cluster_info.txt'
cluster_df = pd.read_csv(cluster_csv, sep = '\t', encoding='iso-8859-1', index_col = None)
cluster_df.index = cluster_df.cluster.astype(str)

relv_arts = (docs.group.value_counts() * cluster_df.rating_bool).sum()
print("Articles removed: {}".format(data.shape[0] - relv_arts))

###
#   Get Relevant articles
###
get_cluster_bool = lambda x: cluster_df.query('cluster == @x').rating_bool.values[0]
mask = np.ones(docs.shape[0], bool)
irrelevant_arts = docs.group.apply(get_cluster_bool) ^ True
irrelevant_idx = docs[irrelevant_arts].index.values
mask[irrelevant_idx] =  False
docs_relv = docs[mask].copy().reset_index()
docs_relv.rename({'index':'orig_idx'}, axis = 1, inplace=True)
docs_relv.drop('Unnamed: 0', axis = 1, inplace=True)

relevant_docs = 'relevant_docs.txt'
docs_relv.to_csv(relevant_docs, sep = '\t', encoding='iso-8859-1', index = False)

get_cluster_rating = lambda x: cluster_df.query('cluster == @x').rating.values[0]
docs_relv['group_rating'] = docs_relv.group.apply(get_cluster_rating)

reading_list = {}
for idx in cluster_df.index:
    c_rating = cluster_df.loc[idx, 'rating']
    if c_rating == 0:
        continue
    else:
        to_read = eval_cluster(docs, dtm_lsa, idx)
        reading_list[idx] = (c_rating, to_read)

valid_clusters = [cluster_df.loc[idx, 'cluster'] for idx in cluster_df.index if cluster_df.loc[idx, 'rating'] > 0]

list_to_read = [reading_list[str(cluster)] for cluster in valid_clusters]
scaled_list_to_read_num = [int(round(rating * len(arts))) for rating, arts in list_to_read]

to_read = [read[1][:num] for num, read in zip(scaled_list_to_read_num, list_to_read)]

articles = []
for l in to_read:
    articles.extend(l)

ratings = pickle.load(open('ratings.pickle', 'rb'))

rate_vals = [0, .25, .5, .75, 1]
ratings_map = {rate_float:rate_vals.index(rate_float) for rate_float in rate_vals}
ratings_label = [ratings_map[rating] for rating in ratings]

articles_idx = [docs[docs.Title == art].index[0] for art in articles]
docs['art_rating'] = 99
docs.loc[articles_idx, 'art_rating'] = ratings_label


