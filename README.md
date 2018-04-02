
## Dataset

### Introduction

UD English-ESL/TLE is a collection of 5,124 English as a Second Language (ESL) sentences (97,681 words), manually annotated with POS tags and dependency trees in the Universal Dependencies formalism. Each sentence is annotated both in its original and error corrected forms. The annotations follow the standard English UD guidelines, along with a set of supplementary guidelines for ESL. The dataset represents upper-intermediate level adult English learners from 10 native language backgrounds, with over 500 sentences for each native language. The sentences were randomly drawn from the Cambridge Learner Corpus First Certificate in English (FCE) corpus. The treebank is split randomly to a training set of 4,124 sentences, development set of 500 sentences and a test set of 500 sentences. Further information is available at esltreebank.org

### File format

#### Raw data

Exam questions:
dataset/UD_English-ESL/fce-released-dataset/prompts/[folders]/doc[number].xml

Learner answers:
dataset/UD_English-ESL/fce-released-dataset/dataset/[folders]/doc[number].xml

Each xml file contains the textual answers for 2 exams written by a English learner. The following are the attribute tags:
- language: native language of the learner
- age: age range of the learner
- score: ??

For each exam:
- question_number
- exam_score
- coded_answer: text content of answer (with tags of FCE error codes)

The details of exams and tags of FCE error codes can be found in dataset/UD_English-ESL/fce-released-dataset/dataset/README

#### Labeled data in CoNLL-U format

The labeled dataset is built in CoNLL-U format.

Original sentences:  
- dataset/UD_English-ESL/data/en_esl-ud-train.conllu  
- dataset/UD_English-ESL/data/en_esl-ud-dev.conllu  
- dataset/UD_English-ESL/data/en_esl-ud-test.conllu  

Corrected sentences:  
- dataset/UD_English-ESL/data/corrected/en_cesl-ud-train.conllu  
- dataset/UD_English-ESL/data/corrected/en_cesl-ud-dev.conllu  
- dataset/UD_English-ESL/data/corrected/en_cesl-ud-test.conllu  


The following are the attributes for each word in a sentence:  
['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

- id: index of word in sentence  
- form: word  
- lemma:  
- upostag: POS tag  
- xpostag: POS tag  
- feats:  
- head:  
- deprel:  
- deps:  
- misc:

("_" means null)

Example of the representation of word:  
([('id', 1), ('form', 'I'), ('lemma', '_'), ('upostag', 'PRON'), ('xpostag', 'PRP'), ('feats', None), ('head', 3), ('deprel','nsubj'), ('deps', None), ('misc', None)])




```python
import data_loader
data_loader.main()
```

    # of train data: 4124
    # of dev data: 500
    # of test data: 500
