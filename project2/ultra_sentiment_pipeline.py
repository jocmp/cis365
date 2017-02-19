import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV as GridSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from csv_cleaner import HTMLCleaner

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.data.load('tokenizers/punkt/english.pickle')

stop = stopwords.words('english')

UNCLEAN_DATA = './training_movie_data.csv'


class UltraPipeline:
    @staticmethod
    def tokenizer(text):
        cleaned_text = HTMLCleaner.clean(text)
        tokens = word_tokenize(cleaned_text)
        tokens_without_punctuation = [UltraPipeline.parse_punctuation(token) for token in tokens]
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens_without_punctuation if len(token) > 0]

    @staticmethod
    def parse_punctuation(word):
        return ''.join([character for character in word if character not in string.punctuation])

    @staticmethod
    def run(grid_search=False):
        df = pd.read_csv(UNCLEAN_DATA)

        training_size = 40000

        x = df.loc[:training_size, 'review'].values
        y = df.loc[:training_size, 'sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.125, random_state=17)

        # Perform feature extraction on the text.
        # Hint: Perhaps there are different preprocessors to
        # test?
        tfidf = TfidfVectorizer(tokenizer=UltraPipeline.tokenizer,
                                strip_accents='unicode',
                                stop_words=stop,
                                lowercase=False)

        # LogisticRegressionCV ==> k = 5 or 10
        clf = LogisticRegressionCV(cv=10, random_state=31, n_jobs=-1)
        lr_tfidf = Pipeline([('vect', tfidf), ('clf', clf)])

        if grid_search:
            UltraPipeline.run_grid_search(X_train, y_train)

        # Train the pipline using the training set.
        lr_tfidf.fit(X_train, y_train)

        # Print the Test Accuracy
        test_score = lr_tfidf.score(X_test, y_test)
        print('Test Accuracy: %.3f' % test_score)

        # Save the classifier
        pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))

    @staticmethod
    def run_grid_search(X, y):
        ''' Used to test out parameters '''
        tfidf = TfidfVectorizer(tokenizer=UltraPipeline.tokenizer,
                                strip_accents='unicode',
                                stop_words=stop)
        clf = LogisticRegression(C=1, penalty='l2', random_state=0, solver='sag')

        pipeline = Pipeline([('vect', tfidf), ('clf', clf)])

        grid = dict(vect__tokenizer=[UltraPipeline.tokenizer, None],
                    vect__strip_accents=['unicode', 'ascii'],
                    vect__lowercase=[True, False],
                    vect__stop_words=['english', stop, None],
                    clf__fit_intercept=[True, False],
                    clf__penalty=['l2'],
                    clf__solver=['liblinear'])

        # Run grid search
        grid_search = GridSearch(pipeline, param_grid=grid, n_jobs=-1)

        grid_search.fit(X, y)

        print(grid_search.cv_results_)


if __name__ == '__main__':
    UltraPipeline.run()
