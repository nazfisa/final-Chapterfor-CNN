# Importing the libraries
import numpy as np
import re
import pickle 
import nltk
#import heapq
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import pandas as pd
import csv


from os import listdir
from nltk.corpus import stopwords
from pickle import dump
#from gensim.models import Word2Vec

#Data=pd.read_excel('Restaurant - Copy.xlsx')
Data = pd.read_csv('data1.csv')

#Data= Data[Data['Polarity'] != 'neutral']
#Data= Data[Data['Polarity'] != 'conflict']
#z=Data['Polarity'].values.astype('str')

X=Data.iloc[:,0].values
y=Data.iloc[:,1].values


for data in range(len(y)):
    if(y[data]=='positive'):
        y[data]=1
    elif(y[data]=='negative'):
        y[data]=0


'''
#Storing as pickle Files
with open('x.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#Umpickling the dataset
with open('x.pickle','rb') as f:
    X=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)

'''

with open('stopwords-bn.txt', 'r', encoding='utf8') as bn:
    bangla_stop_words = [line.strip() for line in bn]

corpus = []
for i in range(0, len(X)):
    Data = re.sub(r'\W', ' ', str(X[i]))
    #review = review.lower()
    Data = re.sub(r'^br$', ' ', Data)
    Data = re.sub(r'\s+br\s+',' ',Data)
    Data = re.sub(r'\s+[a-z]\s+', ' ',Data)
    Data = re.sub(r'^b\s+', '', Data)
    Data = ' '.join([word for word in Data.split() if word not in bangla_stop_words])  
    Data = re.sub(r'\s+', ' ', Data)
    Data = re.sub(r'\॥’...-!?:-:-‘’/‘‘’।,', ' ', Data)
    corpus.append(Data) 
 
    
    
# Splitting the dataset into the Training set and Test set
#y=y.astype('int')

from sklearn.model_selection import train_test_split
trainX, testX, trainy, testY = train_test_split(X, y, test_size = 0.20, random_state = 0)

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

save_dataset([trainX,trainy], 'train.pkl')
save_dataset([testX,testY], 'test.pkl')

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model

# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
print(trainX.shape)

# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=10, batch_size=16)
# save the model
model.save('model.h5')

