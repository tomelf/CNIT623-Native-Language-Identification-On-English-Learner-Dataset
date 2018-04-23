import data_loader
import numpy as np
import pandas as pd
import re
import os.path

from itertools import product
from string import ascii_lowercase

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet

def main():
    cols = ["dim_{}".format(i+1) for i in range(300)] + ["label"] + ["meta_id"]
    train_df = pd.read_csv("dataset/data_doc2vec_non-pretrained/train_docvec_lang_sentid.csv", names=cols)
    dev_df = pd.read_csv("dataset/data_doc2vec_non-pretrained/dev_docvec_lang_sentid.csv", names=cols)
    test_df = pd.read_csv("dataset/data_doc2vec_non-pretrained/test_docvec_lang_sentid.csv", names=cols)
    
    train_actual = train_df["label"].tolist()
    dev_actual = dev_df["label"].tolist()
    test_actual = test_df["label"].tolist()
    
    train_data_df = train_df.iloc[:,:(len(cols)-2)]
    dev_data_df = dev_df.iloc[:,:(len(cols)-2)]
    test_data_df = test_df.iloc[:,:(len(cols)-2)]
    
    print("Start pipeline")
    pipeline = Pipeline([
        ('clf', SVC(decision_function_shape='ovo', C=20)),
    ])
    
    param_grid = dict(
        clf__kernel=[ 'linear', 'poly', 'rbf'],
        clf__C=[0.1, 1, 10, 20])
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10, return_train_score=True)
    
    train_pred = grid_search.fit(train_data_df, train_actual).predict(train_data_df)
    test_pred = grid_search.predict(test_data_df)
    
    with open("task3_test_doc2vec.txt", "w") as output:
        output.write("Training results:\n{}\n".format(classification_report(train_actual, train_pred)))
        output.write("Testing results:\n{}\n".format(classification_report(test_actual, test_pred)))
    print(classification_report(train_actual, train_pred))
    print(classification_report(test_actual, test_pred))
    
    pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_doc2vec.csv")

if __name__ == "__main__":
    main()
