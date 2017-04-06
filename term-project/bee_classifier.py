import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.font_manager

DATA_SAMPLE = './data/last_%d000_weight.csv'
DATA_2000 = './data/last_2000_weight.csv'
ALL_DATA = './data/all_attrs.csv'


class BeeClassifier(object):
    @staticmethod
    def run():
        clf = BeeClassifier()
        scale_samples = pd.read_csv(DATA_SAMPLE % (100))

        X = scale(scale_samples)

        clf.kmeans(X)

    def kmeans(self, x):
        reduced_data = PCA(n_components=2).fit_transform(x)
        kmeans = KMeans(n_clusters=2, n_init=10)

        X_train, X_test = train_test_split(reduced_data, test_size=0.5, random_state=17)

        kmeans.fit(X_train)

        y_predicted_from_training = kmeans.predict(X_train)
        y_predicted_from_test = kmeans.predict(X_test)

        training_error = y_predicted_from_training[y_predicted_from_training == -1].size
        testing_error = y_predicted_from_test[y_predicted_from_test == -1].size

        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='green')
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet')

        plt.show()


if __name__ == '__main__':
    BeeClassifier.run()
