# Databricks notebook source
# MAGIC %md # Functions for used for Scoring

# COMMAND ----------

# MAGIC %md ## init function - Used to initiate Web service

# COMMAND ----------

def init():
  global alan
  global embedding_matrix
  global word_index
  global bestsvc
  global bestrf
  global bestgp 
  global besten 
  global tokenizer
  global code
  global labels

  global inputs_dc, prediction_dc
  from sklearn.externals import joblib
  import xgboost
  import pickle
  import time

  # load the model from file into a global object
  global model
  tokenizer,bestsvc,bestrf,bestgp,besten,code,labels = joblib.load('model.pkl')
  print(time.asctime( time.localtime(time.time()) ) + ": " + str(labels))

  from keras.preprocessing.text import Tokenizer
  from keras.preprocessing.sequence import pad_sequences
  
  # variables
  MAX_SEQUENCE_LENGHT = 10
  MAX_NB_WORDS = 1000
  EMBEDDING_DIM = 100
  VALIDATION_SPLIT = 0.2
  
  import urllib.request
  import numpy as np

  from azure.storage.blob import BlockBlobService, PublicAccess

  print(time.asctime( time.localtime(time.time()) ) + ": Initialisation started")

  block_blob_service = BlockBlobService(account_name='spearfishstorage', account_key='WZG1kIIrAmERvlI23eY3z/IKpPvT+pvf4eA9rgRk+TvAzBHIV/qHV0K7NalPgsq6WLo2c8WVFnifHjlhWEbfzQ==')

  print(time.asctime( time.localtime(time.time()) ) + ": Download Glove from Blobstore")

  block_blob_service.get_blob_to_path("embeddings", "glove.6B.100d.txt","glove.6B.100d.txt")
  embeddings_index={}
  f = open('glove.6B.100d.txt', encoding = 'utf8')
 
  print(time.asctime( time.localtime(time.time()) ) + ": Glove File Downloaded")

  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype = 'float32')
      embeddings_index[word] = coefs
  f.close()

  word_index = tokenizer.word_index
  print("Found %s word vectors." % len(embeddings_index))
  num_words = min(MAX_NB_WORDS, len(word_index))
  embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector

  print(time.asctime( time.localtime(time.time()) ) + ": Initialisation completed")
  return(0)


# COMMAND ----------

# MAGIC %md ## Run function - Function for Scoring Service

# COMMAND ----------

def run():
    import time
    import json
    print(time.asctime( time.localtime(time.time()) ) + ": Start Run Function")
    from sklearn import preprocessing
    import numpy as np
        
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    # variables
    MAX_SEQUENCE_LENGHT = 10
    MAX_NB_WORDS = 1000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    try:
      sequences = tokenizer.texts_to_sequences([str1]) 
      data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGHT)
      embedded_sequence = np.empty(0)
      for input_word in data:
         embedded_sequence = np.append(embedded_sequence, embedding_matrix[input_word])
      
      rfo=bestrf.predict([embedded_sequence]) 
      svco=bestsvc.predict([embedded_sequence]) 
      gpo=bestgp.predict([embedded_sequence]) 
      eno=besten.predict([embedded_sequence]) 

      print(time.asctime( time.localtime(time.time()) ) + ": Score SVC value " + str(svco))
      print(time.asctime( time.localtime(time.time()) ) + ": Score XGB value " + str(gpo))
      print(time.asctime( time.localtime(time.time()) ) + ": Score RF value " + str(rfo))
      print(time.asctime( time.localtime(time.time()) ) + ": Score EN value " + str(eno))

      print(time.asctime( time.localtime(time.time()) ) + ": End Run Function")
      print("SVC:"+labels[svco])
      print("RF:"+labels[rfo])
      print("XG:"+labels[gpo])
      print("EN:"+labels[eno])
      return(json.dumps({"String" : str1,"Code" : code, "svc" : str(svco),"xgb" : str(gpo) ,"rf": str(rfo),"en": str(eno),"svc_l" : str(labels[svco][0]),"xgb_l" : str(labels[gpo][0]) ,"rf_l": str(labels[rfo][0]),"en_l": str(labels[eno][0])}))

    except Exception as e:
        print("Error: {0}",str(e))
        return (str(e))