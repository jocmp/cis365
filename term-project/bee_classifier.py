import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import time

WITH_JUST_WEIGHT = './data/messages_just_weight.csv'
WITH_COORDINATES = './data/messages_with_coords.csv'

class BeeClassifier(object):
    @staticmethod
    def run(dataset, transform=False):
        bee_clf = BeeClassifier()
        samples = pd.read_csv(dataset).sample(n=5000)
        if transform:
            samples = PCA(n_components=2).fit_transform(samples)

        bee_clf.set_title(dataset, transform, len(samples))
        bee_clf.kmeans_scatter_plot(scale(samples))

    def set_title(self, dataset_name, transform, sample_size):
        title = ""
        if (dataset_name == WITH_JUST_WEIGHT):
            title = "Weight with Occurrence Time\n "
        if (dataset_name == WITH_COORDINATES):
            title = "Weight, Coordinates with Occurence Time\n"

        if (transform):
            title += "using 2 components\n"

        title += time.strftime('%l:%M%p %z on %b %d, %Y') + ", n=" + str(sample_size)
        plt.title(title)

    def kmeans_scatter_plot(self, X):
        clf = MiniBatchKMeans(n_clusters=2)
        clf.fit(X)

        centroids = clf.cluster_centers_

        labels = clf.labels_
        colors = ['g.', 'r.']

        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], colors[labels[i]])

        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

        plt.show()

    @staticmethod
    def run_on_individual_hive(hive_id):
        """
        :param hive_id: 
        """
        clf = BeeClassifier()
        clf.plot_hive_weight(clf.hive_data_frame(hive_id))



if __name__ == '__main__':
    BeeClassifier.run_on_individual_hive(49)
