from keras.preprocessing import text
from keras.models import Sequential,Model
from keras.layers import Dense, Activation
from keras.layers import Merge,Flatten
from keras.layers import Convolution1D,Convolution2D,MaxPooling2D,GlobalMaxPooling2D
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
import pickle
import os, os.path, sys
from keras.layers import Input, Embedding
from keras import backend as K
import collections
import random
import re
from keras.utils.visualize_util import plot
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping 

normal_dataset = "/home/elnaz/Documents/k-test/dataset_normal" #input('Please enter the normal dataset file: ')
simple_dataset = "/home/elnaz/Documents/k-test/dataset_simple"#input('Please enter the simple dataset file: ')
words =list(pickle.load(open("/home/elnaz/Documents/k-test/all_words.p",'rb')))
dictionary = dict((words[i],i) for i in range(0,len(words)))

nb_filter = 128
batch_size = 128
iteration = 10


def main():
    normal, simple = embedding_layer()
    label = label_vectorize() 
    build_model(normal,simple,label)
    print(normal.shape)
    print(simple.shape)
    print(label.shape)


def label_vectorize():
    label_tmp = open("/home/elnaz/Documents/k-test/labels",'r').readlines()
    labels = []
    for l in label_tmp:
        label_vec = np.zeros(2)
        label_vec[int(l)] = 1
        labels.append(label_vec)
    labels=np.asarray(labels)
    return labels

def vectorize(line):
    tmp = re.sub(r'[^a-zA-Z0-9 ]','', line)
    word_list = tmp.lower().split()
    line_vec = []
    for w in word_list:
        line_vec.append(int(dictionary.get(w)))
    return line_vec

def embedding_layer():
    dictionary = dict((words[i],i) for i in range(0,len(words)))
    normal_mapped = []
    simple_mapped = []
    normal_lines = open(normal_dataset).readlines()
    simple_lines = open(simple_dataset).readlines()
    for l in normal_lines:
        normal_mapped.append(vectorize(l))
    for l in simple_lines:
        simple_mapped.append(vectorize(l))
    
    max_len = max(len(max(normal_mapped,key=len)), len(max(simple_mapped,key=len)))
    normal_pad = pad_sequences(normal_mapped, maxlen = max_len, dtype = 'int32', padding = 'post',truncating='post',value=-1)
    simple_pad = pad_sequences(simple_mapped, maxlen = max_len, dtype = 'int32', padding = 'post',truncating='post',value=-1)
    return normal_pad, simple_pad

def build_model(normal,simple,label):
    history = open("/home/elnaz/Documents/k-test/history3_4",'w')
#    history.write('left_branch = Sequential()\n left_conv_3 = Sequentialaa\n left_conv_3.add(Convolution2D(nb_filter, 3,1, border_mode = 'same',activation = 'sigmoid', input_shape = (1,147,1), name = 'left_3'))\nleft_conv_3.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))\nleft_conv_4 = Sequential()\nleft_conv_4.add(Convolution2D(nb_filter, 4,1, border_mode = 'same',activation = 'sigmoid', input_shape=(1,147,1), name='left_4'))\nleft_conv_4.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))\nleft_conv_5 = Sequential()\nleft_conv_5.add(Convolution2D(nb_filter, 5,1, border_mode = 'same',activation= 'sigmoid', input_shape = (1,147,1), name = 'left_5'))\nleft_conv_5.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))\nmerge_left = Merge([left_conv_3, left_conv_4, left_conv_5],mode = 'concat')\nleft_branch.add(merge_left)\nleft_branch.add(Dense(128, activation = 'tanh'))\nright_branch = Sequential()\nright_conv_3 = Sequential()\nright_conv_3.add(Convolution2D(nb_filter, 3,1, border_mode ='same',activation= 'sigmoid', input_shape = (1,147,1), name = 'right_3'))\nright_conv_3.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))\nright_conv_4 = Sequential()\nright_conv_4.add(Convolution2D(nb_filter, 4,1, border_mode ='same',activation= 'sigmoid', input_shape=(1,147,1), name='right_4'))\nright_conv_4.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))\nright_conv_5 = Sequential()\nright_conv_5.add(Convolution2D(nb_filter, 5,1, border_mode ='same',activation= 'sigmoid', input_shape = (1,147,1), name = 'right_5'))\nright_conv_5.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))\nmerge_right = Merge([right_conv_3, right_conv_4, right_conv_5],mode ='concat')\nright_branch.add(merge_right)\nleft_branch.add(Dense(128, activation = 'tanh'))\nmerged = Merge([left_branch, right_branch], mode = 'concat')\nmodel = Sequential()\nmodel.add(merged)\nmodel.add(Dense(256, activation = 'tanh'))\nmodel.add(Dense(128, activation='tanh'))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='tanh'))\n model.add(Dense(2,activation = 'softmax'))\nmodel.compile(optimizer = adam,loss = categorical_crossentropy, metrics = [accuracy])\nnormal = normal.reshape((175522,1,147,1))\nsimple = simple.reshape((175522, 1, 147,1))\ncallbacks = [EarlyStopping(monitor=val_loss, min_delta=0.05, verbose=1, patience=3)]\nmodel.fit([normal,normal,normal,simple,simple,simple],label, nb_epoch = 200,batch_size=64, callbacks=callbacks)\n')
    left_branch = Sequential()
    left_conv_3 = Sequential()
    left_conv_3.add(Convolution2D(nb_filter, 3,1, border_mode = 'same',activation = 'sigmoid', input_shape = (1,147,1), name = 'left_3'))
    left_conv_3.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))
    left_conv_4 = Sequential()
    left_conv_4.add(Convolution2D(nb_filter, 4,1, border_mode = 'same',activation = 'sigmoid', input_shape=(1,147,1), name='left_4'))
    left_conv_4.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))
    left_conv_5 = Sequential()
    left_conv_5.add(Convolution2D(nb_filter, 5,1, border_mode = 'same',activation= 'sigmoid', input_shape = (1,147,1), name = 'left_5'))
    left_conv_5.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))
    merge_left = Merge([left_conv_3, left_conv_4, left_conv_5],mode = 'concat')
    left_branch.add(merge_left)
    left_branch.add(Dense(128, activation = 'tanh'))

    right_branch = Sequential()
    right_conv_3 = Sequential()
    right_conv_3.add(Convolution2D(nb_filter, 3,1, border_mode ='same',activation= 'sigmoid', input_shape = (1,147,1), name = 'right_3'))
    right_conv_3.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))
    right_conv_4 = Sequential()
    right_conv_4.add(Convolution2D(nb_filter, 4,1, border_mode ='same',activation= 'sigmoid', input_shape=(1,147,1), name='right_4'))
    right_conv_4.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))
    right_conv_5 = Sequential()
    right_conv_5.add(Convolution2D(nb_filter, 5,1, border_mode ='same',activation= 'sigmoid', input_shape = (1,147,1), name = 'right_5'))
    right_conv_5.add(MaxPooling2D(pool_size=(5,1),dim_ordering="th"))
    merge_right = Merge([right_conv_3, right_conv_4, right_conv_5],mode ='concat')
    right_branch.add(merge_right)
    left_branch.add(Dense(128, activation = 'tanh'))

    merged = Merge([left_branch, right_branch], mode = 'concat')


    model = Sequential()
    model.add(merged)
#    model.add(Flatten)
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Flatten())
#    model.add(Dense(128, activation='tanh'))
    model.add(Dense(2,activation = 'softmax'))
    model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    normal = normal.reshape((175522,1,147,1))
    simple = simple.reshape((175522, 1, 147,1))

#    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.05, verbose=1, patience=5)]

    model.fit([normal,normal,normal,simple,simple,simple],label, nb_epoch = 200,batch_size=64)
    plot(model, to_file='/home/elnaz/Documents/k-test/model3_4.png',show_shapes =True)




main()
