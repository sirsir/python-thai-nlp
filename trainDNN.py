# -*- coding: utf-8 -*-

#~~~ CMD
#~ set PYTHONIOENCODING=utf8
#~ py -3 -c "import sys;print(sys.stdout.encoding)"
#~ python trainDNN.py > output/output_redirected.txt 2>&1

import os

from copy import deepcopy

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 

import pickle


MODE='run'
# MODE='train'


# import word2vec_gensim

# word2vec_gensim.py.simple_demo()

# word2vec_gensim.process_wiki()
#!! python test1.py "data/thwiki-latest-pages-articles.xml.bz2" "output/wiki.th.text"
# word2vec_gensim.process_wiki("data/thwiki-latest-pages-articles.xml.bz2","output/wiki.th.text")

#== test
from gensim.models import Word2Vec
from pythainlp.tokenize import word_tokenize

import numpy as np

#~ create model
# a = ['ฉันรักภาษาไทยเพราะฉันเป็นคนไทยและฉันเป็นคนไทย' ,'ฉันเป็นนักเรียนที่ชื่นชอบวิทยาศาสตร์และเทคโนโลยี' ,'ฉันไม่ใช่โปรแกรมเมอร์เพราะฉันทำมากกว่าคิดเขียนพัฒนาโปรแกรมทดสอบโปรแกรม','ฉันชื่นชอบวิทยาศาสตร์ชอบค้นคว้าตั้งสมมุติฐานและหาคำตอบ']
# b = [list(word_tokenize(i)) for i in a] # ทำการตัดคำแล้วเก็บใน list จะได้เป็น [['ฉัน',...],['ฉัน',...]...]
# model = Word2Vec(b, min_count=1)

#~ load model
# model = Word2Vec.load("downloaded/th/th.bin")
# aa=model.similar_by_word('เป็น')
# print(aa)
# vector = model.wv['ไทย']

def loadW2VModel(binPath):
  # model = Word2Vec.load(binPath, binary=True)
  w2vModel = Word2Vec.load(binPath)
  return w2vModel

def loadW2VModelAndAddVocab(oldModelPath, binPath):
  # model = Word2Vec.load(binPath, binary=True)
  w2vModel = Word2Vec.load(oldModelPath)

  
  # w2vModel = loadW2VModel(oldModelPath)

  tokenses = getTokensesFromPandas(dataSample, "sentence")

  print(tokenses)
  w2vModel = trainExistingModelWord2Vec(w2vModel,tokenses)
  w2vModel.save(f'{NEW_W2V_MODEL_DIR}word2vec_new.model')  # save the model

  return w2vModel

def trainExistingModelWord2Vec(w2vModel, tokenses):
  for tokens in tokenses:
    for word in tokens:
      if word not in w2vModel.wv:
        w2vModel.wv[word] = np.random.uniform(-0.25,0.25,w2vModel.vector_size )  
  return w2vModel

def trainExistingModelWord2Vec3(w2vModel, tokenses):
  model1 = Word2Vec(tokenses, min_count=1)

  w2vModel.reset_from(model1)


def trainExistingModelWord2VecOld(w2vModel, tokenses):
  oldModel = deepcopy(w2vModel)
  print(len(w2vModel.wv.vocab))

  # w2vModel.build_vocab(tokenses, update=True)
  # w2vModel.train(tokenses, total_examples=w2vModel.corpus_count, epochs=w2vModel.iter, min_count=1)

  model1 = Word2Vec(tokenses, min_count=1)

  # model1.reset_from(w2vModel)
  # w2vModel = model1
  w2vModel.reset_from(model1)
  # w2vModel = model1

  for m in ['oldModel', 'w2vModel','model1']:
    print('The vocabulary size of the', m, 'is', len(eval(m).wv.vocab))

  

def bow(word,words):
  return [(1 if w == word else 0) for w in words]

def removeNoVocab(words):
  print(words)
  filtered = [w for w in words if w in w2vModel.wv]

  print('No words:>')
  print([w for w in words if w not in w2vModel.wv])

  return filtered

def getVector(sentence):
  # print('getVector')
  words = word_tokenize(sentence)
  # words = word_tokenize(sentence, engine='icu')
  # vectors = map(lambda x:model.wv(x), words)

  # print('tokenized')
  # print(words)

  # words= list(map(lambda v: removeNoVocab(v), words))
  # words= [removeNoVocab(w) for w in words]
  # words= removeNoVocab(words)

  # print('removed No Vocabs')
  # print(words)

  vectors = []
  for w in words:
    if w in w2vModel.wv:
      vectors.append(w2vModel.wv[w])
  
  if len(vectors) == 0:
    return []

  # vectors= list(map(lambda v: removeNoVocab(v), vectors))

  # print("vector=")
  # print(vectors)
  
  npArray = np.array(vectors)
  avg = np.mean(npArray, axis=0)
  
  return avg

def trainDNN():
  import tflearn
  import tensorflow as tf

  # reset underlying graph data
  tf.reset_default_graph()
  # Build neural network
  # net = tflearn.input_data(shape=[None, len(train_x[0])])
  net = tflearn.input_data(shape=[None, w2vModel.vector_size])
  net = tflearn.fully_connected(net, 8)
  net = tflearn.fully_connected(net, 8)
  # net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
  net = tflearn.fully_connected(net, len(intents), activation='softmax')
  net = tflearn.regression(net)

  # Define model and setup tensorboard
  modelDNN = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
  # Start training (apply gradient descent algorithm)
  modelDNN.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
  
  print('Saving model')
  modelDNN.save(f'{NEW_DNN_MODEL_DIR}modelDNN.tflearn')


  # save all of our data structures
  # import pickle
  pickle.dump( { 'intents':intents, 'train_x':train_x, 'train_y':train_y}, open( "output/training_data", "wb" ) )

  #~~ TEST
  # test = train_x[0]
  # print('test:')

  # results = modelDNN.predict([test])

  # print(results)

  #~~ TEST
  test = dataSample.iloc[2]['sentence']
  print(test)
  print('test:'+test)

  # results = modelDNN.predict([getVector(test)])
  print(modelDNN)
  response = getIntents(test, modelDNN)

  print(response)

  return modelDNN

def getIntents(input, model):
  
  # print('dddd')
  # print(model)
  ERROR_THRESHOLD = 0.25

  vector2dnn = getVector(input)

  if len(vector2dnn) == 0:
    return []

  modelOut = model.predict([vector2dnn])[0]

  # filter out predictions below a threshold
  results = [[i,r] for i,r in enumerate(modelOut) if r>ERROR_THRESHOLD]
  # sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  
  for r in results:
    intent = intents[r[0]]
    accuracy = str(r[1])
    answer = ""
    # answer = df.loc[df['intent'] == intent ]['response'].item()
    # print(answer)
    return_list.append({'intent':intent, 'accuracy': accuracy , 'answer': answer})
    
  # return tuple of intent and probability

  # print(return_list.encode('utf'))
  # return json.dumps(return_list)
  return return_list




def testSavedModel():
  import tflearn
  import tensorflow as tf

    # reset underlying graph data
  tf.reset_default_graph()
  # Build neural network
  # Build neural network
  # net = tflearn.input_data(shape=[None, len(train_x[0])])
  net = tflearn.input_data(shape=[None, w2vModel.vector_size])
  net = tflearn.fully_connected(net, 8)
  net = tflearn.fully_connected(net, 8)
  # net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
  net = tflearn.fully_connected(net, len(intents), activation='softmax')
  net = tflearn.regression(net)

  modelDNN = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
  
  modelDNN.load(f'{NEW_DNN_MODEL_DIR}modelDNN.tflearn')

  test = dataSample.iloc[0]['sentence']
  print(test)
  print('test:'+test)
  
  # results = modelDNN.predict([getVector(test)])
  response = getIntents(test, modelDNN)

  print(response)

  return modelDNN



def testVectorize():
  vector = w2vModel.wv['ไทย']
  print(vector)

  vectors = getVector('ฉันรักภาษาไทยเพราะฉันเป็นคนไทยและฉันเป็นคนไทย')
  print(vector)

def getTokensesFromPandas(dataSample, columnName):
  tokenses = []
  for index, row in dataSample.iterrows():
    tokens = word_tokenize(row[columnName])
    tokenses.append(tokens)

  return tokenses



# =======================================================
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 

NEW_W2V_MODEL_DIR = "models/word2vec/"
NEW_DNN_MODEL_DIR = "models/DNN/"
NEW_DATA4TRAIN_FILE = "models/data4train/training_data.pickle"

dataSample = pd.read_csv("data/KERRY_dataset - intent2sentence.csv") 
dataResponse = pd.read_csv("data/KERRY_dataset - intent2response.csv") 
# Preview the first 5 lines of the loaded data 
# dataSample.head()

# print(dataSample.head())
intents = set()
if MODE=='train':
  

  for index, row in dataSample.iterrows():
    intents.add(row['intent'])

  intents = list(intents)
  print(intents)
else:
  pickleObj = pickle.load(open(f'{NEW_DATA4TRAIN_FILE}','rb'))
  intents = pickleObj['intents']
#============ LOAD word2vec, train, save

# Prepare Dir
if not os.path.exists(NEW_W2V_MODEL_DIR):
    os.makedirs(NEW_W2V_MODEL_DIR)


#======= Comment this and rerun  to train DNN !?!

if MODE=='train':
    w2vModel = loadW2VModelAndAddVocab("downloaded/th/th.bin",f'{NEW_W2V_MODEL_DIR}word2vec_new.model')
else:
  w2vModel = None
  w2vModel = loadW2VModel(f'{NEW_W2V_MODEL_DIR}word2vec_new.model')

# 


#================= TRAIN NEURAL
# create train and test lists
train_x = []
train_y = []


for index, row in dataSample.iterrows():
  print('Add training data:'+str(index))
  print(row)
  train_x.append(getVector(row['sentence']))
  train_y.append(bow(row['intent'],intents))

# train_x = np.array(train_x).reshape(-1, 32, 32, 3)
train_x = np.array(train_x)
train_y = np.array(train_y)

if MODE=='train':
  pickle.dump( {'intents':intents, 'train_x':train_x, 'train_y':train_y}, open( f'{NEW_DATA4TRAIN_FILE}', "wb" ) )


print('sample training data (in&out)')
print(train_x[0])
print(train_y[0])

if MODE=='train':
  modelDNN=trainDNN()
else:
  modelDNN=testSavedModel()


#=== TEST MORE
test = 'เมื่อไหร่จะถึง'
print(test)
print('test:'+test)

# results = modelDNN.predict([getVector(test)])
response = getIntents(test, modelDNN)
print(response)

test = 'สวัสดี'
print(test)
print('test:'+test)

# results = modelDNN.predict([getVector(test)])
response = getIntents(test, modelDNN)
print(response)

test = 'ฉฮ'
print(test)
print('test:'+test)

# results = modelDNN.predict([getVector(test)])
response = getIntents(test, modelDNN)
print(response)