#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df = pd.read_csv(url)


# In[2]:


X = df.drop('Class', axis=1)
y = df['Class']


# In[3]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)


# In[26]:


#using stratified sampling

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seed
np.random.seed(42)

# Generate dataset with 1000 samples and 20 features
#X, y = make_classification(n_samples=1000, n_features=20)

# Calculate sample size using formula
p = 0.5  # proportion of positive cases
z = 1.96  # z-score for 95% confidence level
d = 0.05  # margin of error
n = (z**2 * p * (1-p)) / (d**2)
n = int(np.ceil(n))
print(f"Sample size: {n}")

# Split data into five stratified samples
skf = StratifiedKFold(n_splits=5)
samples = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    samples.append((X_train, y_train))

# Train and evaluate five different models on each sample
models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB()]
for i, (X_sample, y_sample) in enumerate(samples):
    print(f"Sample {i+1}:")
    for model in models:
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        acc = accuracy_score(y_sample, y_pred)
        print(f"{type(model).__name__}: {acc:.3f}")
    print()


# In[25]:


#using random sampling

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import numpy as np

# Set random seed
np.random.seed(42)

# Generate dataset with 1000 samples and 20 features
# X, y = make_classification(n_samples=1000, n_features=20)

# Calculate sample size using formula
p = 0.5  # proportion of positive cases
z = 1.96  # z-score for 95% confidence level
d = 0.05  # margin of error
n = (z**2 * p * (1-p)) / (d**2)
n = int(np.ceil(n))
print(f"Sample size: {n}")

# Split data into five random samples
samples = []
for i in range(5):
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=n)
    samples.append((X_sample, y_sample))

# Train and evaluate five different models on each sample
models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB()]
for i, (X_sample, y_sample) in enumerate(samples):
    print(f"Sample {i+1}:")
    for model in models:
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        acc = accuracy_score(y_sample, y_pred)
        print(f"{type(model).__name__}: {acc:.3f}")
    print()


# In[24]:


# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import numpy as np

# Set random seed
np.random.seed(42)

# Generate dataset with 1000 samples and 20 features
#X, y = make_classification(n_samples=1000, n_features=20)

# Create 10 clusters based on the first 100 samples
clusters = np.repeat(np.arange(10), 10)
cluster_sizes = [np.sum(clusters == i) for i in range(10)]
print(f"Cluster sizes: {cluster_sizes}")

# Create five cluster samples
samples = []
for i in range(5):
    # Sample two clusters without replacement
    cluster_sample = np.random.choice(np.arange(10), size=2, replace=False)
    # Sample all samples within the selected clusters
    idx = [np.where(clusters == c)[0] for c in cluster_sample]
    idx = np.concatenate(idx)
    X_sample, y_sample = X[idx], y[idx]
    samples.append((X_sample, y_sample))

# Train and evaluate five different models on each sample
models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB()]
for i, (X_sample, y_sample) in enumerate(samples):
    print(f"Sample {i+1}:")
    for model in models:
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        acc = accuracy_score(y_sample, y_pred)
        print(f"{type(model).__name__}: {acc:.3f}")
    print()


# In[29]:


from imblearn.over_sampling import SMOTE
import numpy as np

# Set random seed
np.random.seed(42)

# Generate dataset with 772 samples and 31 features
X, y = make_classification(n_samples=772, n_features=31, weights=[0.9, 0.1])

# Calculate sample size using formula
p = 0.5  # proportion of positive cases
z = 1.96  # z-score for 95% confidence level
d = 0.05  # margin of error
n = (z**2 * p * (1-p)) / (d**2)
n = int(np.ceil(n))
print(f"Sample size: {n}")

# Split data into five samples with SMOTE oversampling
samples = []
for i in range(5):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    samples.append((X_train, X_test, y_train, y_test))

# Train and evaluate five different models on each sample
models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB()]
for i, (X_train, X_test, y_train, y_test) in enumerate(samples):
    print(f"Sample {i+1}:")
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{type(model).__name__}: {acc:.3f}")
    print()



# In[28]:


from imblearn.over_sampling import ADASYN
import numpy as np

# Set random seed
np.random.seed(42)

# Generate dataset with 772 samples and 31 features
X, y = make_classification(n_samples=772, n_features=31, weights=[0.9, 0.1])

# Calculate sample size using formula
p = 0.5  # proportion of positive cases
z = 1.96  # z-score for 95% confidence level
d = 0.05  # margin of error
n = (z**2 * p * (1-p)) / (d**2)
n = int(np.ceil(n))
print(f"Sample size: {n}")

# Generate 5 stratified samples with ADASYN oversampling
samples = []
for i in range(5):
    X_sample, y_sample = X, y
    ada = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)
    X_resampled, y_resampled = ada.fit_resample(X_sample, y_sample)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    samples.append((X_train, X_test, y_train, y_test))

# Train and evaluate five different models on each sample
models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB()]
for i, (X_train, X_test, y_train, y_test) in enumerate(samples):
    print(f"Sample {i+1}:")
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{type(model).__name__}: {acc:.3f}")
    print()

