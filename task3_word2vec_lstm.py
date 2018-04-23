import data_loader
import numpy as np
import pandas as pd
import pickle
import os
import nltk
import re
import timeit

from torch.autograd import Variable
import torch

from sklearn import preprocessing, svm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score

from util.classification.lstm_nli import LSTMNLI, LSTMNLIPOS

def main():
    meta_list, data_list = data_loader.load_data(load_train=True, load_dev=True, load_test=True)

    train_meta, train_meta_corrected, \
    dev_meta, dev_meta_corrected, \
    test_meta, test_meta_corrected = meta_list

    train_data, train_data_corrected, \
    dev_data, dev_data_corrected, \
    test_data, test_data_corrected = data_list

    label_to_ix_list = train_meta["native_language"].unique().tolist()
    label_to_ix = dict([(l,i+1) for i,l in enumerate(label_to_ix_list)])
    
    for model_type in ["wordseqs", "posseqs"]:
        if model_type=="wordseqs":
            # word sequences
            X_train = [[d["form"].tolist(), label_to_ix[train_meta["native_language"].iloc[i]]] for i,d in enumerate(train_data)]
            X_dev = [[d["form"].tolist(), label_to_ix[dev_meta["native_language"].iloc[i]]] for i,d in enumerate(dev_data)]
            X_test = [[d["form"].tolist(), label_to_ix[test_meta["native_language"].iloc[i]]] for i,d in enumerate(test_data)]
            EMBEDDING_DIM = 300
            HIDDEN_DIM = 200
            LABEL_DIM = len(label_to_ix)
            model = LSTMNLI(EMBEDDING_DIM, HIDDEN_DIM, LABEL_DIM, bidirectional=False)
        if model_type=="posseqs":
            # POS sequences
            X_train = [[d["upostag"].tolist(), label_to_ix[train_meta["native_language"].iloc[i]]] for i,d in enumerate(train_data)]
            X_dev = [[d["upostag"].tolist(), label_to_ix[dev_meta["native_language"].iloc[i]]] for i,d in enumerate(dev_data)]
            X_test = [[d["upostag"].tolist(), label_to_ix[test_meta["native_language"].iloc[i]]] for i,d in enumerate(test_data)]
            pos_to_ix = dict()
            for i in range(len(X_train)):
                sent = X_train[i][0]
                for pos in sent:
                    if pos not in pos_to_ix:
                        pos_to_ix[pos] = len(pos_to_ix)
            if "_" not in pos_to_ix:
                pos_to_ix["_"] = len(pos_to_ix)
            EMBEDDING_DIM = len(pos_to_ix)
            HIDDEN_DIM = len(pos_to_ix)
            LABEL_DIM = len(label_to_ix)
            model = LSTMNLIPOS(EMBEDDING_DIM, HIDDEN_DIM, LABEL_DIM, pos_to_ix, bidirectional=True)
            
        model.cuda()
        model.set_train_data(X_train)
        # model.set_dev_data(X_dev)
        model.train(epoch=1000, lr=0.5)
        # torch.save(model, 'model_{}.pt'.format(model_type))

        with open("task3_doc2vec_lstm_results_{}.txt".format(model_type), "w") as output:
            output.write("Evaluate train performance\n")
            preds = []
            actuals = []
            for i in range(len(X_train)):
                actual = X_train[i][1]
                pred = model.test(X_train[i][0])
                _, pred = torch.max(model.test(X_train[i][0]), 1)

                preds.append(int(pred.data.cpu()))
                actuals.append(actual)
            preds = [label_to_ix_list[(p-1)] for p in preds]
            actuals = [label_to_ix_list[(p-1)] for p in actuals]
            output.write("Train report:\n {}\n".format(classification_report(actuals, preds)))
            output.write("Accuracy: {}\n".format(accuracy_score(actuals, preds)))

            output.write("Evaluate test performance\n")
            preds = []
            actuals = []
            for i in range(len(X_test)):
                actual = X_test[i][1]
                pred = model.test(X_test[i][0])
                _, pred = torch.max(model.test(X_test[i][0]), 1)

                preds.append(int(pred.data.cpu()))
                actuals.append(actual)
            preds = [label_to_ix_list[(p-1)] for p in preds]
            actuals = [label_to_ix_list[(p-1)] for p in actuals]
            output.write("Testing report:\n {}\n".format(classification_report(actuals, preds)))
            output.write("Accuracy: {}\n".format(accuracy_score(actuals, preds)))

if __name__ == "__main__":
    main()
