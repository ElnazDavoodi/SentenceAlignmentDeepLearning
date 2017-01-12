# Sentence Alignment using Siamese Deep Network

Sentence alignment is an area of research in Natural Language Processing (NLP) which has application in Machine Translation, Summarization, Simplification, etc. The goal in sentence alignment is to find mappings between sentences across parallel texts. This mapping is not necessary a one-to-one mapping between sentences of two parallel texts. 

In this project, we use a Siamese Deep Network to do sentence alignment. Previously we used traditional NLP methods to do sentence alignment across monolingual parallel texts. In the traditional methods, we used Term Frequency-Inverse Document Frequency (TF-IDF) in order to find alignment across monolingual parallel texts. In this approach, we assumed the flow of information is the same in both sides of monolingual parallel texts (i.e. normal side vs. simple side). Thus, we can restrict the scope of search for finding a mapping for a given sentence in the one side of a parallel text from the entire aligned document, to a paragraph or two in the other side of the parallel texts. This means, to find an alignment for a sentence which occurred in ith paragraph (P<sub>i</sub>), we considered only paragraphs P<sub>min(0, i-j)</sub> till paragraph P<sub>min(last, i+j)</sub>. 

Instead of using traditional methods, we were interested to investigate if deep networks can be applied to sentence alignment. Thus we created a dataset from a monolingual parallel corpus (the Simple English Wikipedia corpus) to be used in this task.


##Dataset
The Simple English Wikipedia corpus is a parallel corpus contains two versions: 1) the document-aligned and 2) the sentence-aligned versions. The dataset is created using both versions (see create_dataset.py).
