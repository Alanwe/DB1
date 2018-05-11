# Databricks notebook source
# MAGIC %md
# MAGIC ### Version 4 : Operationalise Models 

# COMMAND ----------

# MAGIC %md If required copy fully_categorized.csv to local filesystem

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.gocomparestorage.blob.core.windows.net",
  "eakEcVfZFuJcAwe8k5stWFrjjHC2zmI3y0jfHC9MYB0lvvHZhmiV5nvfFXqS56G0PArOy1e2pB1Xvz8Iw6uq9Q==")
dbutils.fs.ls("wasbs://blobgocompare@gocomparestorage.blob.core.windows.net/")
#dbutils.fs.cp("wasbs://blobgocompare@gocomparestorage.blob.core.windows.net/fully_categorized.csv","/tmp")
dbutils.fs.cp("/tmp/fully_categorized.csv","file:/tmp/")
dbutils.fs.cp("/tmp/glove.6B.100d.txt","file:/tmp/")
#dbutils.fs.cp("file:/tmp/glove.6B.100d.txt","/tmp")

# COMMAND ----------

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/tmp/fully_categorized.csv')


# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lt /tmp/fully_categorized.csv
# MAGIC ls -lt /tmp/glove.6B.100d.txt

# COMMAND ----------

#%sh
#wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.6B.zip
#cp glove.6B.100d.txt /tmp/

# COMMAND ----------

# MAGIC %md Load Python packages, ensure Keras and Tensorflow are attached to cluster

# COMMAND ----------

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier

# imports for random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

# imports for transforming text data to features
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# measure how long the script takes to run
from datetime import datetime
from spark_sklearn import GridSearchCV

# COMMAND ----------

# MAGIC %md Create EnsembleClassifier Class and balanced_subsample Function

# COMMAND ----------

class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):

        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):


        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg
      
def balanced_subsample(y, size=None):
    subsample = []
      
    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))
    print(n_smp)
    
    for label in y.value_counts().index:
        sample = y[y==label].index.values
        index_range = range(sample.shape[0])
        indexes = np.random.choice(index_range, size = n_smp, replace = False)
        subsample.extend(indexes)
        
    return(subsample)

# COMMAND ----------

# MAGIC %md Load from CSV to DataFrame

# COMMAND ----------

startTime = datetime.now()

# variables
MAX_SEQUENCE_LENGHT = 10
MAX_NB_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# read in merged data
print('start reading parquet')
print('start reading parquet')
all_data =pd.read_csv(r'/tmp/fully_categorized.csv', index_col=0, encoding='ISO-8859-1')
# remove rows where raw transaction description contains nothing as it breaks the model
all_data2= all_data.loc[all_data['Raw_Transaction_Description'].notnull(),:]
print('reduce data size')


# COMMAND ----------

startTime = datetime.now()

# variables
MAX_SEQUENCE_LENGHT = 10
MAX_NB_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# read in merged data
print('start reading parquet')
all_data =pd.read_csv(r'/tmp/fully_categorized.csv', index_col=0, encoding='ISO-8859-1')
# remove rows where raw transaction description contains nothing as it breaks the model
all_data= all_data.loc[all_data['Raw_Transaction_Description'].notnull(),:]
print('reduce data size')


# COMMAND ----------

bal_data = balanced_subsample(all_data['Category'], size=8500)
all_data = all_data.iloc[bal_data,:]


# tokenize raw transaction description
print('start preprocessing')
# force category and raw transaction description columns into string type
all_data['Category'] = all_data['Category'].astype(str)
all_data['Raw_Transaction_Description'] = all_data['Raw_Transaction_Description'].astype(str)

texts = all_data['Raw_Transaction_Description'].values

# convert string categories to numeric categories
le = preprocessing.LabelEncoder()
le.fit(all_data['Category'].values)
labels = le.transform(all_data['Category'].values)

# tokenize raw transaction description
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGHT)

print('Shape of data tensor: ', data.shape)
print('Shape of label tensor: ', labels.shape)

# COMMAND ----------

#print(word_index)
#print(tokenizer)

# COMMAND ----------

print(texts)

# COMMAND ----------

#load in glove embeddings
embeddings_index={}
f = open(r'/tmp/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# COMMAND ----------

# MAGIC %md Create embedding matrix

# COMMAND ----------

num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #words not found in embedding index will be all-zeros
        embedding_matrix[i] = embedding_vector

# COMMAND ----------

# MAGIC %md 
# MAGIC Create embedding layer using embedding matrix as weights
# MAGIC All that the Embedding layer does is to map the integer inputs to the vectors 
# MAGIC found at the corresponding index in the embedding matrix, i.e. the 
# MAGIC sequence [1, 2] would be converted to [embeddings[1], embeddings[2]]. </BR>
# MAGIC This means that the output of the Embedding layer will be a 3D tensor of 
# MAGIC shape (samples, sequence_length, embedding_dim).

# COMMAND ----------

embedded_data = np.zeros([len(data),EMBEDDING_DIM*MAX_SEQUENCE_LENGHT])
for input_row_nr,input_row in enumerate(data):
    embedded_sequence = np.empty(0)
    for input_word in input_row:
        embedded_sequence = np.append(embedded_sequence, embedding_matrix[input_word])
    embedded_data[input_row_nr] = embedded_sequence

# COMMAND ----------

# MAGIC %md Split the data into training and validation set

# COMMAND ----------

sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=0)
[index_split] = sss.split(data, labels)
x_train, x_val = embedded_data[index_split[0]], embedded_data[index_split[1]]
y_train, y_val = labels[index_split[0]], labels[index_split[1]]

# COMMAND ----------

print(y_train.shape)

# COMMAND ----------

# MAGIC %md Create SVC Model

# COMMAND ----------

from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
parameters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'), 'C':[1, 20]}
svr = svm.SVC()
clf = GridSearchCV(sc, svr, param_grid=parameters,scoring='accuracy')
clf.fit(x_train, y_train)
print(clf.best_params_ )
bestsvc = clf.best_estimator_
print(clf.best_score_ )

# COMMAND ----------

# MAGIC %md Create Random Forest Model

# COMMAND ----------

en_rf = RandomForestClassifier(n_estimators=64, max_depth=32, min_samples_split=128,
                             random_state=0)

# COMMAND ----------

parametersRF = {"max_depth": [2,3,4,None],
              "max_features": [1, 3, 7,8,9,10,11,12],
              "min_samples_split": [2, 3, 10, 16],
              "min_samples_leaf": [1, 2, 3, 4],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
parametersRF = {"max_depth": [None],
              "max_features": [9],
              "min_samples_split": [2],
              "min_samples_leaf": [1],
              "bootstrap": [False],
              "criterion": ["gini"]}

ST= datetime.now()
clfRF = GridSearchCV(sc, en_rf, param_grid=parametersRF,n_jobs=5,scoring='accuracy')
clfRF.fit(x_train, y_train)
ET= datetime.now()
print("Model Time Taken",(ET - ST))
print("Best Accuracy",clfRF.best_score_ )
print("Best Params",clfRF.best_score_ )
print(clfRF.best_params_ )
bestrf = clf.best_estimator_

# COMMAND ----------

# MAGIC %md Create XGBoost Model

# COMMAND ----------

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
xgb_model = xgb.XGBClassifier()

XGBparameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6,7,8],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5,50,500], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

XGBparameters = {'learning_rate': [0.05], 'subsample': [0.8], 'objective': ['binary:logistic'], 'seed': [1337], 'max_depth': [6], 'silent': [1], 'colsample_bytree': [0.7], 'nthread': [4], 'min_child_weight': [11], 'n_estimators': [500], 'missing': [-999]}

ST= datetime.now()
clfXGB = GridSearchCV(sc,xgb_model, param_grid=XGBparameters, n_jobs=5, scoring='accuracy',verbose=2, refit=True)
clfXGB.fit(x_train, y_train)
ET= datetime.now()
print("Model Time Taken",(ET - ST))
print("Best Accuracy",clfXGB.best_score_ )
print("Best Params",clfXGB.best_score_ )
print(clfXGB.best_params_ )
bestgp = clfXGB.best_estimator_

# COMMAND ----------

endTime= datetime.now()
delta = endTime - startTime
print("Time Taken",delta)

# COMMAND ----------

# MAGIC %md ##Operationalisation(O16N)

# COMMAND ----------

# MAGIC %md ### Scoring function

# COMMAND ----------

# Initialize the deployment environment
def ScoreInit():
  global alan
  global embedding_matrix
  # variables
  MAX_SEQUENCE_LENGHT = 10
  MAX_NB_WORDS = 1000
  EMBEDDING_DIM = 100
  VALIDATION_SPLIT = 0.2
  alan =98
  #load in glove embeddings

  import urllib.request
  import numpy as np

  f = urllib.request.urlopen("https://worksheets.codalab.org/rest/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/contents/blob/")
  embeddings_index={}
  #f = open(r'/tmp/glove.6B.100d.txt', encoding = 'utf8')
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype = 'float32')
      embeddings_index[word] = coefs
  f.close()
  word_index = tokenizer.word_index

  print('Found %s word vectors.' % len(embeddings_index))
  num_words = min(MAX_NB_WORDS, len(word_index))
  embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          #words not found in embedding index will be all-zeros
          embedding_matrix[i] = embedding_vector

def Score(str1):
    from sklearn import preprocessing
    import numpy as np
        
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    try:
      tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
      tokenizer.fit_on_texts(texts)
      sequences = tokenizer.texts_to_sequences([str])      
      word_index = tokenizer.word_index
      data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGHT)
      embedded_data = np.zeros([len(data),EMBEDDING_DIM*MAX_SEQUENCE_LENGHT])
      for input_row_nr,input_row in enumerate(data):
          embedded_sequence = np.empty(0)
          for input_word in input_row:
              embedded_sequence = np.append(embedded_sequence, embedding_matrix[input_word])
          embedded_data[input_row_nr] = embedded_sequence
      return embedded_data

    except Exception as e:
        print("Error: {0}",str(e))
        return (str(e))


# COMMAND ----------

ScoreInit()

# COMMAND ----------


r=Score("Credit Card Payment")
print(r)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Deploy Model

# COMMAND ----------

import pickle
#m = [word_index,bestsvc,bestrf,bestgp]
m = [tokenizer,bestsvc,bestrf,bestgp]
fh = open(b"/tmp/model.pkl","wb")
pickle.dump(m,fh)

# COMMAND ----------

from azure.storage.blob import BlockBlobService, PublicAccess
block_blob_service = BlockBlobService(account_name='spearfishstorage', account_key='WZG1kIIrAmERvlI23eY3z/IKpPvT+pvf4eA9rgRk+TvAzBHIV/qHV0K7NalPgsq6WLo2c8WVFnifHjlhWEbfzQ==')
   
block_blob_service.create_blob_from_path("models","TaxtClassify2" ,"/tmp/model.pkl")