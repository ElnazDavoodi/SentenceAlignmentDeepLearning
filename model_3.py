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






def sentence_to_vector(map_word, map_vec, sent):
    sent = sent.replace("b'","")
    sent = sent.replace("'","")
    words = sent.split()
    vec = np.empty((0,300), dtype = 'float')
    for w in words:
        vec=np.append(vec,np.asarray(map_vec[map_word[w]]), axis = 0)

#    print(vec.shape)
    return vec
#    print(vec.shape)
#    print(type(vec))



def __init__():
    map_word_id, map_id_vec = embedding_layer()
    count = 0
    for f in os.listdir("/home/elnaz/Documents/k-test/normal/"):
        output_normal = open("/media/elnaz/My Passport/backup/embedded_normal/"+f,'wb')
        output_simple = open("/media/elnaz/My Passport/backup/embedded_simple/"+f,'wb')
        vec_n_text = ""
        vec_s_text = ""
        normal=str(open(os.path.join("/home/elnaz/Documents/k-test/normal/",f),'r').read())
        vec_n = sentence_to_vector(map_word_id, map_id_vec, normal)
        simple=str(open(os.path.join("/home/elnaz/Documents/k-test/simple/",f),'r').read())
        vec_s = sentence_to_vector(map_word_id, map_id_vec, simple)
        np.savetxt(output_normal, np.asarray(vec_n))
        np.savetxt(output_simple,np.asarray(vec_s))
        output_normal.close()
        output_simple.close()



def build_model_old():
    nb_filter = 128
    filter_length = [3]
    conv = []
    history = open("/home/elnaz/Documents/k-test/history3_1",'w')
    history.write('nb_filter = 128, filter_length = 3,4,5 , shuffle data\n 1D globalmaxpooling \n convolution: same,activation= sigmoid')

    left_branch = Sequential()
    left_conv_3 = Sequential()
    left_conv_3.add(Convolution1D(nb_filter, 3, border_mode = 'same',activation = 'sigmoid', input_shape = (195, 300), name = 'left_3'))
    left_conv_3.add(GlobalMaxPooling1D())
    left_conv_4 = Sequential()
    left_conv_4.add(Convolution1D(nb_filter, 4, border_mode = 'same',activation = 'sigmoid', input_shape=(195,300), name='left_4'))
    left_conv_4.add(GlobalMaxPooling1D())
    left_conv_5 = Sequential()
    left_conv_5.add(Convolution1D(nb_filter, 5, border_mode = 'same',activation= 'sigmoid', input_shape = (195,300), name = 'left_5'))
    left_conv_5.add(GlobalMaxPooling1D())
    merge_left = Merge([left_conv_3, left_conv_4, left_conv_5],mode = 'concat')
    left_branch.add(merge_left)
#    left_branch.add(Convolution1D(nb_filter,i,border_mode='same',activation='sigmoid',input_shape=(195,300), name = str(i)+"_convolution"))
#        left_branch.add(GlobalMaxPooling1D())
#        left_branch.add(MaxPooling1D(pool_length = i, name =str(i)+"_maxpooling"))
#        left_branch.add(Dense(64, activation = 'tanh'))



    right_branch = Sequential()
    right_conv_3 = Sequential()
    right_conv_3.add(Convolution1D(nb_filter, 3, border_mode ='same',activation= 'sigmoid', input_shape = (195, 300), name = 'right_3'))
    right_conv_3.add(GlobalMaxPooling1D())
    right_conv_4 = Sequential()
    right_conv_4.add(Convolution1D(nb_filter, 4, border_mode ='same',activation= 'sigmoid', input_shape=(195,300), name='right_4'))
    right_conv_4.add(GlobalMaxPooling1D())
    right_conv_5 = Sequential()
    right_conv_5.add(Convolution1D(nb_filter, 5, border_mode ='same',activation= 'sigmoid', input_shape = (195,300), name = 'right_5'))
    right_conv_5.add(GlobalMaxPooling1D())
    merge_right = Merge([right_conv_3, right_conv_4, right_conv_5],mode ='concat')
    right_branch.add(merge_right)
#    for i in filter_length:
#        right_branch.add(Convolution1D(nb_filter,i,border_mode='same',activation='sigmoid',input_shape=(195,300),name = str(i)+"_convolution"))
#        right_branch.add(GlobalMaxPooling1D())
#        right_branch.add(MaxPooling1D(pool_length = i,name=str(i)+"_maxpooling"))
#        right_branch.add(Dense(64, activation = 'tanh'))

#    right_branch.add(Dense(128, input_shape = (195,300)))
    merged = Merge([left_branch, right_branch], mode = 'concat')


    model = Sequential()
    model.add(merged)
    model.add(Dense(128, activation = 'tanh'))
#    model.add(Flatten())
    model.add(Dense(2,activation = 'softmax'))
    model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])

    labels = open("/home/elnaz/Documents/k-test/labels",'r').readlines()
    count = 0
    batch_size = 128
    iteration = 10
    iter_count = 0
    files = os.listdir("/media/elnaz/My Passport/backup/embedded_normal/")
    while iter_count<iteration:
        acc = 0
        count = 0
        batch_count = 0
        while count<len(files):
            normal_batch = []
            simple_batch = []
            label_batch = []
            while len(normal_batch)<batch_size:
                normal=open(os.path.join("/media/elnaz/My Passport/backup/embedded_normal/",files[count%len(files)]),'r').readlines()
#            print(files[count%len(files)])
                simple=open(os.path.join("/media/elnaz/My Passport/backup/embedded_simple/",files[count%len(files)]),'r').readlines()
                try:
                    mtx_normal = [l.split(" ") for l in normal]
#            flat_normal = [val for sublist in mtx for val in sublist]
                    vec_normal = np.reshape(np.asarray(mtx_normal),(195,300))
                    mtx_simple = [l.split(" ") for l in simple]
#            flat = [val for sublist in mtx for val in sublist]
                    vec_simple = np.reshape(np.asarray(mtx_simple),(195,300))
                    normal_batch.append(vec_normal)
                    simple_batch.append(vec_simple)
                    label_tmp = np.zeros(2)
                    label_tmp[int(labels[int(files[count%len(files)])])] = 1
                    label_batch.append(label_tmp)
#                    print(files[count%len(files)])
#                print(files[count%len(files)])
#                print(normal_batch)
#                print('----------------------------------')
#                print(simple_batch)
#                print('===================================')
#                print(label_batch)
#                c = list(zip(normal_batch, simple_batch, label_batch))
#                random.shuffle(c)
#                normal_batch, simple_batch, label_batch = zip(*c)
#                print('*********************new normal_batch----------')
#                print(normal_batch)
#                print('********************new simple_batch************')
#                print(simple_batch)
#                print('********************new labels***************')
#                print(label_batch)
                except ValueError:
                    os.remove(os.path.join("/media/elnaz/My Passport/backup/embedded_normal/",str(files[count%len(files)])))
                    os.remove(os.path.join("/media/elnaz/My Passport/backup/embedded_simple/",str(files[count%len(files)])))
                    print('error :'+str(files[count%len(files)]))
                finally:
                    count+=1
#                print(np.asarray(label_batch))

            c = list(zip(normal_batch, simple_batch, label_batch))
            random.shuffle(c)
            normal_batch = [e[0] for e in c]
            simple_batch = [e[1] for e in c]
            label_batch = [e[2] for e in c]
            batch_count +=1
#        normal_batch, simple_batch, label_batch = zip(*c)



            hist=model.train_on_batch([np.asarray(normal_batch),np.asarray(normal_batch),np.asarray(normal_batch),np.asarray(simple_batch),np.asarray(simple_batch),np.asarray(simple_batch)],np.asarray(label_batch))
            acc+=hist[1]
            print('iter: ', iter_count,' batch count ',batch_count,' batch acc :', hist[1])
        print('batch_count ',batch_count, 'accuracy: ', acc, float(acc/batch_count))
        history.write('iteration :'+str(iter_count)+' '+model.metrics_names[1]+' '+str(float(float(acc)/float(batch_count)))+'\n')
        print(hist)
#        print('loss: ',float(loss/batch_size))
        iter_count+=1
    plot(model, to_file='/home/elnaz/Documents/k-test/model3_1.png',show_shapes =True)

#    normal_file = open("/home/elnaz/Documents/k-test/data_normal",'r').readlines()
#    simple_file = open("/home/elnaz/Documents/k-test/data_simple",'r').readlines()
#    end = len(normal_file)
#    batch_size = 2
#    batch_count = 0
#    normal_batch = []
#    simple_batch = []
#    while start<end-step:
#        print('start', start)
#        print('end',end)
#        batch_count = 0
#        while batch_count<batch_size:
#            vec = normal_file[start:start+step]
#            mtx = [v.split(" ") for v in vec]
#            flat_vec = [val for sublist in mtx for val in sublist]
#            print(len(flat_vec))
#            vec_normal = np.reshape(np.asarray(flat_vec),(155,300))
#            vec = simple_file[start:start+step]
#            mtx = [v.split(" ") for v in vec]
#            flat_vec = [val for sublist in mtx for val in sublist]
#            vec_simple = np.reshape(np.asarray(flat_vec),(155,300))
#            normal_batch.append(vec_normal)
#            simple_batch.append(vec_simple)
#            batch_count +=1
#            start +=step
#        print('normal batch',np.asarray(normal_batch).shape)
#        print('simple_batch',np.asarray(simple_batch).shape)
#        y = np.ones(batch_size)
#        print(y.shape)
#        hist=model.fit([np.asarray(normal_batch), np.asarray(simple_batch)],y,nb_epoch=1, batch_size=batch_size)
#        print(hist.history)
#        plot(model, to_file='/home/elnaz/Documents/k-test/model.png',show_shapes = True)


#score = model.evaluate(data_1, labels, batch_size = 16)
#p)arint(score)

#embedding_layer()
#__init__()
#build_model()
