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

from util.feature.CharNGrams import CharNGrams
from util.feature.FunctionWords import FunctionWords
from util.feature.ItemSelector import ItemSelector

def main():
    meta_list, data_list = data_loader.load_data(load_train=True, load_dev=False, load_test=True)
    train_meta, train_meta_corrected, \
        test_meta, test_meta_corrected = meta_list
    train_data, train_data_corrected, \
        test_data, test_data_corrected = data_list
    
    train_data_df = [[" ".join(td["form"].tolist()), " ".join(td["upostag"].tolist())] for td in train_data]
    test_data_df = [[" ".join(td["form"].tolist()), " ".join(td["upostag"].tolist())] for td in test_data]
    train_data_df = pd.DataFrame(train_data_df, columns=["form", "upostag"])
    test_data_df = pd.DataFrame(test_data_df, columns=["form", "upostag"])
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
                        ('selector', ItemSelector(key='form')),
                        ('tfidf', TfidfVectorizer(analyzer="word", 
                                                  ngram_range=(1,4), 
                                                  lowercase=True, 
                                                  vocabulary=funcwords_vocabs)
                        ),
                        ('reduce_dim', TruncatedSVD()),
                ])),
                ('charngrams', Pipeline([
                        ('selector', ItemSelector(key='form')),
                        ('tfidf', TfidfVectorizer(analyzer="char", 
                                                  ngram_range=(3,3), 
                                                  lowercase=True, 
                                                  vocabulary=char_ngrams_vocabs)
                        ),
                        ('reduce_dim', TruncatedSVD()),
                ])),
                ('wordngrams', Pipeline([
                        ('selector', ItemSelector(key='form')),
                        ('tfidf', TfidfVectorizer(analyzer="word", 
                                                  ngram_range=(1,2), 
                                                  lowercase=True)
                        ),
                        ('reduce_dim', TruncatedSVD()),
                ])),
                ('posngrams', Pipeline([
                        ('selector', ItemSelector(key='upostag')),
                        ('tfidf', TfidfVectorizer(analyzer="word", 
                                                  ngram_range=(2,2), 
                                                  lowercase=True)
                        )
                ])),
            ],
            transformer_weights={
                'funcwords': 1,
                'charngrams': 1,
                'wordngrams': 1,
                'posngrams': 1,
            },
        )),
        ('clf', SVC(decision_function_shape='ovo', C=20)),
    ])
    
    param_grid = dict(union__funcwords__reduce_dim__n_components=[10, 50, 100],
                      union__charngrams__reduce_dim__n_components=[50, 100, 200],
                      union__wordngrams__reduce_dim__n_components=[50, 100, 200],
                      union__wordngrams__tfidf__ngram_range=[(1,1), (2,2), (1,2)],
                      union__posngrams__tfidf__ngram_range=[(1,1), (2,2), (3,3), (1,3)],
                      clf__kernel=['linear', 'poly', 'rbf'],
                      clf__C=[0.1, 1, 10, 20])
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10, return_train_score=True)
    
    train_pred = grid_search.fit(train_data_df, train_actual).predict(train_data_df)
    test_pred = grid_search.predict(test_data_df)
    
    # with open("task3_test_funcwords.txt", "w") as output:
    # with open("task3_test_charngrams.txt", "w") as output:
    # with open("task3_test_wordngrams.txt", "w") as output:
    # with open("task3_test_posngrams.txt", "w") as output:
    with open("task3_test_all.txt", "w") as output:
        output.write("Training results:\n{}\n".format(classification_report(train_actual, train_pred)))
        output.write("Testing results:\n{}\n".format(classification_report(test_actual, test_pred)))
    print(classification_report(train_actual, train_pred))
    print(classification_report(test_actual, test_pred))
    
    # pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_funcwords.csv")
    # pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_charngrams.csv")
    # pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_wordngrams.csv")
    # pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_posngrams.csv")
    pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_all.csv")

if __name__ == "__main__":
    main()
