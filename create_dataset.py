import os, sys


'''prepare_dataset() method create a dataset using the sentence aligned version of the Simple English Wikipedia corpus. In this method, we take already aligned sentences from the normal and simple sides of the Simple English Wikipedia corpus and add them to the data set as true aligned sentences, so we give them the label of 1 in the label file. Also, we take one sentence before and after of each aligned simple sentence in the simple side of the document aligned Simple English Wikipedia corpus and add it to the dataset with label 0. This means that the pair of ''a normal sentence in the sentence alignment'' and ''one sentence before and after its true alignment'' creates a negative instance in the dataset. Thus it is added, where applicable, with label 0. As a result, the data set contains two files, one is dataset_noram and the other one is dataset_simple, the sentences in each line make a pair of (normal,  simple). We also create a label file, each line contains a 0 or 1 which shows if that line of data set is a true alignment or a false alignment.'''

directory=raw_input('Please enter the directory to create the data set and label files: ')
def prepare_dataset():
'''simple.aligned file contains the simple side of the simple english wikipedia corpus'''
    simp_a = open("/home/elnaz/Documents/k-test/SEW/sentence-aligned.v2/simple.aligned",'r').readlines()
'''normal.aligned file contains the normal side of the simple english wikipedia corpus'''    
    norm_a = open("/home/elnaz/Documents/k-test/SEW/sentence-aligned.v2/normal.aligned",'r').readlines()

'''simple.txt contains the simple part of document aligned side of the simple english wikipedia corpus'''
    simp = open("/home/elnaz/Documents/k-test/SEW/document-aligned.v2/simple.txt",'r').readlines()

'''output_n: file to dump the normal sentences of dataset and output_s: file to dump the simple sentences of the dataset'''
    output_n = open(directory+"/dataset_normal",'a')
    output_s = open(directory+"/dataset_simple",'a')
    target = open(directory+"/labels",'a')
#    output_n = open("/home/elnaz/Documents/k-test/dataset_normal",'a')
#    output_s = open("/home/elnaz/Documents/k-test/dataset_simple",'a')
#    target = open("/home/elnaz/Documents/k-test/labels",'a')
    file_content = dict()
    for line in simp:
        vals = line.split("\t")
        if vals[0].strip().lower().rstrip() in file_content:
            file_content[vals[0].strip().lower()].append(vals[2].rstrip())
        else:
            array = []
            array.append(vals[2].rstrip())
            file_content[vals[0].strip().lower()] = array
    for i in range(0,len(simp_a)):
        a_n = norm_a[i]
        a_s = simp_a[i]
        output_n.write(a_n.split("\t")[2])
        output_s.write(a_s.split("\t")[2])
        target.write(str(1)+"\n")
        vals = a_s.split("\t")
        txt = vals[0].strip().lower().rstrip()
        if txt in file_content:
            output_n.write(a_n.split("\t")[2])
            sents=a_s.split("\t")[2].rstrip().split(' . ')
            if len(sents)>1:
                search_f = sents[0].strip()+" ."
                search_l = sents[-1].strip()
            elif len(sents) == 1:
                search_f = sents[0]
                search_l = sents[0]
            try:
                ent_f = file_content[txt].index(search_f)
                ent_l = file_content[txt].index(search_l)
                if ent_f > 0:
                    output_s.write(file_content[txt][ent_f-1])
                    target.write(str(0)+"\n")
                if ent_l<len(file_content[txt])-1:
                    output_s.write(file_content[txt][ent_l+1])
                    target.write(str(0)+"\n")
            except ValueError:
                print(search_f, search_l)
    output_n.close()
    output_s.close()
    target.close()

def main():
    prepare_dataset()


