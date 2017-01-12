#from keras.preprocessing import text
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.layers import Merge
import numpy as np
#from keras.utils.np_utils import to_categorical
import pandas as pd
import os, sys, os.path
from matplotlib import pyplot as plt
from math import ceil
import pickle
import re

'''
normal_dir and simple_dir is the path to normal_raw and simple_raw files which are created by create_dataset.py
#normal_dir = "/path/to/normal_raw/file"
#simple_dir = "/path/to/simple_raw/file"
'''
normal_dir = raw_input("normal file: ")
simple_dir = raw_input("simple file: ")


'''
length_check method creates a dictionary of [sentence length --> number of instances(cases) of each length] for all instances in training set.
'''
def length_check(directory):
    text = {}
    lines = open(directory,'r').readlines()
    for f in lines:
        tmp = re.sub(r'[^a-zA-Z0-9. ]','', f)
        key=len(tmp.split())
        if key in text:
            text[key] +=1
        else:
            text[key] = 1
    return text

'''
data_trimming method gets two dictionaries of [sentence length --> number of sentences of each length],
then merges the two dictionaries to have distribution over sentence lengths. The goal is to remove very small and 
very large sentences from training set which occur rarely. Thus, in this distribution, we cut the first 0.05 and last 0.9 percent of the distribution.
The distribution of sentence lengths before and after this filtering is plotted.
'''
def data_trimming():
    simple_len=length_check(simple_dir)
    total=length_check(normal_dir)
    for k,v in simple_len.items():
        if k in total:
            total[k] +=v
        else:
            total[k] = v
    all_values_original = [v*[k] for k,v in total.items()]
    all_values_original = [item for sublist in all_values_original for item in sublist]

    x = list(total.keys())
    y = list(total.values())
    plt.scatter(x,y)
    plt.ylim(0,max(y))
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.title('distribution of text lengths before filtering')
    plt.show()
    plt.close()
    
    new_keys =list(total.keys())[int(ceil(0.05*len(total))):int(ceil(0.90*len(total)))]
    new_values =list(total.values())[int(ceil(0.05*len(total))):int(ceil(0.90*len(total)))]
    new_total = dict(zip(new_keys,new_values))

    x = list(new_total.keys())
    y = list(new_total.values())
    plt.scatter(x,y)
    plt.ylim([0,max(new_total.values())])
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.title('distribution of text lengths after filtering')
    plt.show()
    plt.close()
    return new_total

'''
dump_words() method calls data_trimming() first to get a dictionary of [sentence length --> frequency of sentences of each length],
then it calls extract_words() to extract the list of words from the instances with a length between the minimum and maximum of accepted length (see data_trimming() method).
Then, the list of all words in the sentences with length between min and max, as a pickle object to be used for later uses.
'''
def dump_words():
    dictionary = data_trimming()
    min_len = min(dictionary.keys())
    max_len = max(dictionary.keys())
    print('maximum length: ',max_len)
    print('minimum length: ', min_len)
    words_set1=list(extract_words(normal_dir, min_len, max_len))
    words_set2 = list(extract_words(simple_dir, min_len, max_len))
    all_words = set(list(set(words_set1) | set(words_set2)))
    pickle.dump(all_words,open('/home/elnaz/Documents/k-test/all_words.p','wb'))



def extract_words(directory,minimum, maximum):
    tokens = []
    if 'normal' in directory:
        out_label = 'normal'
    else:
        out_label = 'simple'
    lines = open(directory, 'r').readlines()
    for i in range(0, len(lines)):
        file_to_read = re.sub(r'[^a-zA-Z0-9 ]','',lines[i])
        file_len = len(file_to_read.split())
        if file_len>= minimum and file_len<=maximum:
#            pad_list = ["pad"]*((int)(maximum)+1-file_len)
#                print(pad_list)
#            pad = " ".join(pad_list)
#            file_to_read = file_to_read+pad
            tokens.extend(file_to_read.split())
#            out_file=open('/home/elnaz/Documents/k-test/'+out_label+"/"+str(i),'w')
#            out_file.write(file_to_read)
#            out_file.close()
#                pickle.dump(file_to_read,open('/home/elnaz/Documents/k-test/'+out_label+'/'+f+'.p','wb'))
    return set(tokens)

def main():
    dump_words()

main()
