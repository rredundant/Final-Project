#!/usr/bin/env python
# coding: utf-8

# In[76]:


import csv
import pandas as pd
import glob
import random
import numpy as np

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

from scipy.sparse import csr_matrix


# ## Query used to get data from Hive to get raw data from Hadoop.

# We ran this in Hadoop on WordPress.com's servers: https://mc.a8c.com/pb/213b7/#plain

# In[62]:


# Each month has a different CSV of data. Here we programatically combine them into one dataframe.

files = glob.glob('Data/*.csv')

li = []

for filename in files:
    num_lines = sum(1 for l in open(filename))
    size = int(num_lines / 6 ) # use these values: 3,4,5,6
    skip_idx = random.sample(range(1, num_lines), num_lines - size)
    df = pd.read_csv(filename, skiprows=skip_idx, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

# Shuffle the data.
df = shuffle(df)

# Confirm size of dataset.
df.shape


# In[63]:


# Check class balance. Looks pretty balanced!

df['plan_purchased_nice'].value_counts()


# In[64]:


# Remove stopwords so as to clean up our features (vectorized text).
stop = set(stopwords.words('english'))
df['msg_whole_clean'] = df['msg_whole'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[65]:


# Create features out of text in chat transcripts.

vectorizer = CountVectorizer(ngram_range=(2, 6), analyzer ='word', max_df =.75, min_df = .05) 

features = vectorizer.fit_transform(df['msg_whole_clean'])


# In[66]:


# Split into train and test segments.

X_train, X_test, y_train, y_test = train_test_split(
         features, df['plan_purchased_nice'], test_size=0.25, random_state=42)


# In[67]:


# Run data through various classifiers to find the highest accuracy.

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    CalibratedClassifierCV(LinearSVC()),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for visual comparison (optional)
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    
    # For the last three classifiers above to run, we need to convert 
    # the sparse matrix generated from the countvectorizer step above
    # into a dense matrix.
    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[80]:


#Start individually optimizing hyperparameters of highest performing algorithm: GradientBoostingClassifier.

# Experiment with different learning rates.
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
print('****Results****')

for eta in learning_rates:
    clf = GradientBoostingClassifier(learning_rate=eta)

    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Learning Rate: {:.4%}".format(eta))
    print("Accuracy: {:.4%}".format(acc))
    print("="*30)


# In[84]:


# Experiment with different n_estimators.

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
print('****Results****')

for estimator in n_estimators:
    clf = GradientBoostingClassifier(n_estimators=estimator)

    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("N estimators: {}".format(estimator))
    print("Accuracy: {:.4%}".format(acc))
    print("="*30)

#  Highest: 32


# In[85]:


# Experiment with different max_depths.

max_depths = np.linspace(1, 32, 32, endpoint=True)

print('****Results****')

for max_depth in max_depths:
    clf = GradientBoostingClassifier(max_depth=max_depth)

    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Max Depth: {}".format(max_depth))
    print("Accuracy: {:.4%}".format(acc))
    print("="*30)


# In[86]:


# Experiment with different min_samples_splits.

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

print('****Results****')

for min_samples_split in min_samples_splits:
    clf = GradientBoostingClassifier(min_samples_split=min_samples_split)
    
    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Max Samples Split: {}".format(min_samples_split))
    print("Accuracy: {:.4%}".format(acc))
    print("="*30)


# In[89]:


# Experiment with different min_samples_leafs.

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

for min_samples_leaf in min_samples_leafs:
    clf = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)

    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Max Samples Leafs: {}".format(min_samples_leaf))
    print("Accuracy: {:.4%}".format(acc))
    print("="*30)


# In[91]:


# Experiment with different max_features.

max_features = list(range(1,features.shape[1]))

for max_feature in max_features:
    clf = GradientBoostingClassifier(max_features=max_feature)
    
    X_train = csr_matrix(X_train).todense()
    X_test = csr_matrix(X_test).todense()
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Max Features: {}".format(max_feature))
    print("Accuracy: {:.4%}".format(acc))
    print("="*30)


# In[94]:


# By combining the individual optimizations above into one algorithm
# we can see the impact this has on accuracy: 42.3077%.
# That's worse than the classifier with NO hyperparameters (see next cell).

clf = GradientBoostingClassifier(learning_rate = 0.5,
                                    n_estimators = 32,
                                    max_features = 12,
                                    min_samples_split = 0.7,
                                    min_samples_leaf = 0.1)
    
X_train = csr_matrix(X_train).todense()
X_test = csr_matrix(X_test).todense()
clf.fit(X_train, y_train)
name = clf.__class__.__name__

train_predictions = clf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))


# In[95]:


# This is the classifier run with no hyperparamters.
# Accuracy = 46.7248%

clf = GradientBoostingClassifier()
    
X_train = csr_matrix(X_train).todense()
X_test = csr_matrix(X_test).todense()
clf.fit(X_train, y_train)
name = clf.__class__.__name__

train_predictions = clf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))


# In[107]:


# Finally, we tested optimizing each of the hyperparamters below
# one by one, adding a new parameter each time to the highest performing
# from the previous run.
# Accuracy = 47.5361%

clf = GradientBoostingClassifier(learning_rate =.5,
                                 n_estimators = 8,
                                 max_depth = 2,
                                 min_samples_split = 0.2,
                                 max_features = 35)
#                                      min_samples_leaf 

X_train = csr_matrix(X_train).todense()
X_test = csr_matrix(X_test).todense()
clf.fit(X_train, y_train)
name = clf.__class__.__name__

train_predictions = clf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))



# In[ ]:




