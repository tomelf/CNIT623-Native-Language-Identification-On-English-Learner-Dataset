
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



### Data Loader

To use the data loader, you need to first install the [CoNLL-U Parser](https://github.com/EmilStenstrom/conllu) built by [Emil Stenström](https://github.com/EmilStenstrom).

The following is an example to use data_loader:


```python
import data_loader

meta_list, data_list = data_loader.load_data(load_train=True, load_dev=True, load_test=True)

train_meta, train_meta_corrected, \
dev_meta, dev_meta_corrected, \
test_meta, test_meta_corrected = meta_list

train_data, train_data_corrected, \
dev_data, dev_data_corrected, \
test_data, test_data_corrected = data_list

f = "csv" # or "json"
data_loader.dump_preprocessed_data(name="train", train_meta, train_data, format=f)
data_loader.dump_preprocessed_data(name="train_corrected", train_meta_corrected, train_data_corrected, format=f)
data_loader.dump_preprocessed_data(name="dev", dev_meta, dev_data, format=f)
data_loader.dump_preprocessed_data(name="dev_corrected", dev_meta_corrected, dev_data_corrected, format=f)
data_loader.dump_preprocessed_data(name="test", test_meta, test_data, format=f)
data_loader.dump_preprocessed_data(name="test_corrected", test_meta_corrected, test_data_corrected, format=f)
```

    Load data (load_train=True, load_dev=True, load_test=True)
    Load raw conllu dataset (load_train=True, load_dev=True, load_test=True)
    Processing dataset/UD_English-ESL/data/en_esl-ud-train.conllu
    Processing dataset/UD_English-ESL/data/corrected/en_cesl-ud-train.conllu
    Processing dataset/UD_English-ESL/data/en_esl-ud-dev.conllu
    Processing dataset/UD_English-ESL/data/corrected/en_cesl-ud-dev.conllu
    Processing dataset/UD_English-ESL/data/en_esl-ud-test.conllu
    Processing dataset/UD_English-ESL/data/corrected/en_cesl-ud-test.conllu
    Build metadata
    Build sentences
    =================
    Dump train to preprocessed/train
    Dump train_corrected to preprocessed/train_corrected
    Dump dev to preprocessed/dev
    Dump dev_corrected to preprocessed/dev_corrected
    Dump test to preprocessed/test
    Dump test_corrected to preprocessed/test_corrected



```python
train_meta.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>doc_id</th>
      <th>sent</th>
      <th>native_language</th>
      <th>age_range</th>
      <th>score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>doc2664</td>
      <td>I was &lt;ns type="S"&gt;&lt;i&gt;shoked&lt;/i&gt;&lt;c&gt;shocked&lt;/c&gt;...</td>
      <td>Russian</td>
      <td>21-25</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>doc648</td>
      <td>I am very sorry to say it was definitely not a...</td>
      <td>French</td>
      <td>26-30</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>doc1081</td>
      <td>Of course, I became aware of her feelings sinc...</td>
      <td>Spanish</td>
      <td>16-20</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>doc724</td>
      <td>I also suggest that more plays and films shoul...</td>
      <td>Japanese</td>
      <td>21-25</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>doc567</td>
      <td>Although my parents were very happy &lt;ns type="...</td>
      <td>Spanish</td>
      <td>31-40</td>
      <td>34.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data[0]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>form</th>
      <th>lemma</th>
      <th>upostag</th>
      <th>xpostag</th>
      <th>feats</th>
      <th>head</th>
      <th>deprel</th>
      <th>deps</th>
      <th>misc</th>
      <th>meta_id</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>I</td>
      <td>_</td>
      <td>PRON</td>
      <td>PRP</td>
      <td>None</td>
      <td>3</td>
      <td>nsubj</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>was</td>
      <td>_</td>
      <td>VERB</td>
      <td>VBD</td>
      <td>None</td>
      <td>3</td>
      <td>cop</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>shoked</td>
      <td>_</td>
      <td>ADJ</td>
      <td>JJ</td>
      <td>None</td>
      <td>0</td>
      <td>root</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>because</td>
      <td>_</td>
      <td>SCONJ</td>
      <td>IN</td>
      <td>None</td>
      <td>8</td>
      <td>mark</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I</td>
      <td>_</td>
      <td>PRON</td>
      <td>PRP</td>
      <td>None</td>
      <td>8</td>
      <td>nsubj</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>had</td>
      <td>_</td>
      <td>AUX</td>
      <td>VBD</td>
      <td>None</td>
      <td>8</td>
      <td>aux</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alredy</td>
      <td>_</td>
      <td>ADV</td>
      <td>RB</td>
      <td>None</td>
      <td>8</td>
      <td>advmod</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spoken</td>
      <td>_</td>
      <td>VERB</td>
      <td>VBN</td>
      <td>None</td>
      <td>3</td>
      <td>advcl</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>with</td>
      <td>_</td>
      <td>ADP</td>
      <td>IN</td>
      <td>None</td>
      <td>10</td>
      <td>case</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>them</td>
      <td>_</td>
      <td>PRON</td>
      <td>PRP</td>
      <td>None</td>
      <td>8</td>
      <td>nmod</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>and</td>
      <td>_</td>
      <td>CONJ</td>
      <td>CC</td>
      <td>None</td>
      <td>8</td>
      <td>cc</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>I</td>
      <td>_</td>
      <td>PRON</td>
      <td>PRP</td>
      <td>None</td>
      <td>14</td>
      <td>nsubj</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>had</td>
      <td>_</td>
      <td>AUX</td>
      <td>VBD</td>
      <td>None</td>
      <td>14</td>
      <td>aux</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>taken</td>
      <td>_</td>
      <td>VERB</td>
      <td>VBN</td>
      <td>None</td>
      <td>8</td>
      <td>conj</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>two</td>
      <td>_</td>
      <td>NUM</td>
      <td>CD</td>
      <td>None</td>
      <td>16</td>
      <td>nummod</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>autographs</td>
      <td>_</td>
      <td>NOUN</td>
      <td>NNS</td>
      <td>None</td>
      <td>14</td>
      <td>dobj</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>.</td>
      <td>_</td>
      <td>PUNCT</td>
      <td>.</td>
      <td>None</td>
      <td>3</td>
      <td>punct</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Dumped files are under ./preprocessed/[name]/. 
- meta.csv: the same format as the above variable "train_meta"
- [number].csv: the same format as the above variable "train_data[0]"

