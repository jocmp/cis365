"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""

import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV as GridSearch
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from os import remove
import string
from csv_cleaner import HTMLCleaner

# Hint: These are not actually used in the current
# pipeline, but would be used in an alternative 
# tokenizer such as PorterStemming.
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.data.load('tokenizers/punkt/english.pickle')

stop = stopwords.words('english')

UNCLEAN_DATA = './training_movie_data.csv'
CLEAN_DATA = './clean_data.csv'


class UltraPipeline:
    @staticmethod
    def tokenizer(text):
        cleaned_text = HTMLCleaner.clean(text)
        tokens = word_tokenize(cleaned_text)
        tokens = [i for i in tokens if i not in string.punctuation]
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    @staticmethod
    def run(grid_search=False):
        df = pd.read_csv(UNCLEAN_DATA)
        #
        # Hint: This might be an area to change the size
        # of your training and test sets for improved
        # predictive performance.
        training_size = 40000

        x = df.loc[:training_size, 'review'].values
        y = df.loc[:training_size, 'sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.0875, random_state=0)

        # Perform feature extraction on the text.
        # Hint: Perhaps there are different preprocessors to
        # test?
        tfidf = TfidfVectorizer(tokenizer=UltraPipeline.tokenizer,
                                strip_accents='unicode',
                                stop_words=stop,
                                lowercase=False)

        # Create a pipeline to vectorize the data and then perform regression.
        # Hint: Are there other options to add to this process?
        # Look to documentation on Regression or similar methods for hints.
        # Possibly investigate alternative classifiers for text/sentiment.
        clf = LogisticRegression(fit_intercept=False, random_state=31, n_jobs=-1)
        lr_tfidf = Pipeline([('vect', tfidf), ('clf', clf)])

        if grid_search:
            UltraPipeline.run_grid_search(X_train, y_train)

        # Train the pipline using the training set.
        lr_tfidf.fit(X_train, y_train)

        # Print the Test Accuracy
        test_score = lr_tfidf.score(X_test, y_test)
        print('Test Accuracy: %.3f' % test_score)

        # Save the classifier for use later.
        pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))

    @staticmethod
    def run_grid_search(X, y):
        tfidf = TfidfVectorizer(tokenizer=UltraPipeline.tokenizer,
                                strip_accents='unicode',
                                stop_words=stop)
        clf = LogisticRegression(C=1, penalty='l2', random_state=0, solver='sag')

        pipeline = Pipeline([('vect', tfidf), ('clf', clf)])

        grid = dict(vect__lowercase=[True, False],
                    clf__fit_intercept=[True, False])

        start = time()
        # run grid search
        grid_search = GridSearch(pipeline, param_grid=grid, n_jobs=-1)

        grid_search.fit(X, y)

        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.cv_results_)))
        print(grid_search.cv_results_)


if __name__ == '__main__':
    UltraPipeline.run(grid_search=False)
