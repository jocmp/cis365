import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn import svm

SHORT_DATA = './data/last_1000_weight.csv'
ALL_DATA = './data/just_weight.csv'
sample_size = 100


class BeeClassifier(object):
    @staticmethod
    def run():
        scale_samples = pd.read_csv(ALL_DATA)

        X = scale(scale_samples)

        print(X.shape)

        BeeClassifier.plot_waveform(X)
        # X_reduced = PCA(n_components=2).fit_transform(X)

        # kmeans = KMeans(n_clusters=2, random_state=17).fit(X_reduced)

        # print(kmeans.labels_)

    @staticmethod
    def plot_waveform(x, sample_size=1000):
        plt.plot(x[0:sample_size])
        plt.xlabel("Sample number")
        plt.ylabel("Weight")
        plt.show()


if __name__ == '__main__':
    BeeClassifier.run()
