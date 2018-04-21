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

from util.feature.CharNGrams import CharNGrams
from util.feature.FunctionWords import FunctionWords

def main():
    meta_list, data_list = data_loader.load_data(load_train=True, load_dev=False, load_test=True)
    train_meta, train_meta_corrected, \
        test_meta, test_meta_corrected = meta_list
    train_data, train_data_corrected, \
        test_data, test_data_corrected = data_list
    
    train_data_df = [" ".join(td["form"].tolist()) for td in train_data]
    test_data_df = [" ".join(td["form"].tolist()) for td in test_data]
    train_actual = train_meta["native_language"].tolist()
    test_actual = test_meta["native_language"].tolist()
    
    print("Start pipeline")
    # char_ngrams
    char_ngrams_md = CharNGrams(3)
    char_ngrams_vocabs = char_ngrams_md.get_features()
    # function words
    funcwords_md = FunctionWords("function_words.txt")
    funcwords_vocabs = funcwords_md.get_features()
    
    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('funcwords', Pipeline([
                        ('tfidf', TfidfVectorizer(analyzer="word", 
                                                  ngram_range=(1,4), 
                                                  lowercase=True, 
                                                  vocabulary=funcwords_vocabs)
                        ),
                        ('reduce_dim', TruncatedSVD(n_components=50)),
                ])),
                ('charngrams', Pipeline([
                        ('tfidf', TfidfVectorizer(analyzer="char", 
                                                  ngram_range=(3,3), 
                                                  lowercase=True, 
                                                  vocabulary=char_ngrams_vocabs)
                        ),
                        ('reduce_dim', TruncatedSVD(n_components=50)),
                ])),
            ],
            transformer_weights={
                'funcwords': 1,
                'charngrams': 1,
            },
        )),
        ('clf', SVC()),
    ])
    
    param_grid = dict(union__funcwords__reduce_dim__n_components=[10, 50, 100],
                      union__charngrams__reduce_dim__n_components=[100, 300, 500],
                      clf__C=[20, 30])
    
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10, return_train_score=True)
    train_pred = grid_search.fit(train_data_df, train_actual).predict(train_data_df)
    test_pred = grid_search.predict(test_data_df)
    print(classification_report(train_actual, train_pred))
    print(classification_report(test_actual, test_pred))
    
    pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search.csv")

if __name__ == "__main__":
    main()
