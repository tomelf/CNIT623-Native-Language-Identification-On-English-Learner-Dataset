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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib

from util.feature.ItemSelector import ItemSelector

def main():
    post_df = data_loader.load_post_metadata()
    # Filter languages with too few data
    post_df = post_df[post_df["native_language"].isin(
        ['Russian', 'French', 'Spanish', 'Japanese', 'Chinese', 
         'Turkish', 'Portuguese', 'Korean', 'German', 'Italian']
    )]
    
    data = [" ".join(essay) for essay in post_df["ans1_errors"].tolist()] + \
        [" ".join(essay) for essay in post_df["ans2_errors"].tolist()]
    actual = post_df["native_language"].tolist() + post_df["native_language"].tolist()
    
    train_data_df, test_data_df, train_actual, test_actual = train_test_split(
        data, actual, test_size=0.2, random_state=1)
    
    print("Start pipeline")
    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('ans1errors', Pipeline([
                        ('tfidf', TfidfVectorizer(analyzer="word", 
                                                  ngram_range=(1,1), 
                                                  norm = 'l1',
                                                  lowercase=True)
                        )
                ])),
            ],
            transformer_weights={
                'ans1errors': 1,
            },
        )),
        ('clf', SVC(decision_function_shape='ovo', C=20)),
    ])
    param_grid = dict(
        union__ans1errors__tfidf__norm=['l1', 'l2', None],
        clf__kernel=['rbf'],
        clf__C=[0.1, 1, 5, 10, 20],
    )
    
    best_model_dump = 'task3_wholetext_best.pkl'
    if os.path.isfile(best_model_dump):
        pipeline = joblib.load(best_model_dump)
        train_pred = pipeline.fit(train_data_df, train_actual).predict(train_data_df)
        test_pred = pipeline.predict(test_data_df)
        
        with open("task3_grid_search_wholetext_errors.txt", "w") as output:
            output.write("Training results:\n{}\n".format(classification_report(train_actual, train_pred)))
            output.write("Testing results:\n{}\n".format(classification_report(test_actual, test_pred)))
        print(classification_report(train_actual, train_pred))
        print(classification_report(test_actual, test_pred))
        
    else:
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10, 
                                   return_train_score=True, scoring='accuracy')
        train_pred = grid_search.fit(train_data_df, train_actual).predict(train_data_df)
        test_pred = grid_search.predict(test_data_df)
        
        with open("task3_grid_search_wholetext_errors.txt", "w") as output:
            output.write("Training results:\n{}\n".format(classification_report(train_actual, train_pred)))
            output.write("Testing results:\n{}\n".format(classification_report(test_actual, test_pred)))
        print(classification_report(train_actual, train_pred))
        print(classification_report(test_actual, test_pred))

        joblib.dump(grid_search.best_estimator_, best_model_dump)
        pd.DataFrame(grid_search.cv_results_).to_csv("task3_grid_search_wholetext_errors.csv")

if __name__ == "__main__":
    main()
