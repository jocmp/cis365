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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
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

PARSED_DATA = './clean_movie_data.csv'
UNCLEAN_DATA = './training_movie_data.csv'


def preprocessor(text):
    return HTMLCleaner.clean(text)


def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stemmer = PorterStemmer()
    return [stemmer.stem(text) for text in tokens]


# Read in the dataset and store in a pandas dataframe
df = pd.read_csv(UNCLEAN_DATA)

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.
# 
# Hint: This might be an area to change the size
# of your training and test sets for improved 
# predictive performance.
training_size = 40000

x = df.loc[:training_size, 'review'].values
y = df.loc[:training_size, 'sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.125, random_state=42)

# Perform feature extraction on the text.
# Hint: Perhaps there are different preprocessors to
# test?
tfidf = TfidfVectorizer(strip_accents='unicode',
                        lowercase=False,
                        stop_words='english',
                        preprocessor=preprocessor,
                        tokenizer=tokenizer)

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in
# sklearn or other model selection strategies.

# param_grid = dict(vect__stop_words=['english', stop], vect__lowercase=[True, False],
#                  vect__strip_accents=['unicode', 'ascii'])

# search = GridSearch(lr_tfidf, param_grid=param_grid, n_jobs=20)
# search.fit(X_train, y_train)
# print('CV results:')
# pprint.pprint(search.cv_results_)

# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.

lr_tfidf = Pipeline(
    [('vect', tfidf), ('clf', LogisticRegression(C=1, fit_intercept=False, penalty='l2', random_state=32392967))])

# Train the pipline using the training set.
lr_tfidf.fit(X_train, y_train)

# Print the Test Accuracy
test_score = lr_tfidf.score(X_test, y_test)
print('Test Accuracy: %.3f' % test_score)

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
