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

DATA_1000 = './data/last_1000_weight.csv'
DATA_2000 = './data/last_2000_weight.csv'
ALL_DATA = './data/all_attrs.csv'
sample_size = 100


class BeeClassifier(object):
    @staticmethod
    def run():
        clf = BeeClassifier()
        scale_samples = pd.read_csv(DATA_2000)

        X = scale(scale_samples)

        # clf.plot_waveform(X)
        # clf.plot_kmeans(X)
        # clf.plot_gaussian_mixture(X)
        clf.plot_svm(X)

    def plot_waveform(self, x, sample_size=1000):
        plt.plot(x[0:sample_size])

        plt.xlabel("Sample number")
        plt.ylabel("Weight")

        plt.show()

    def plot_kmeans(self, x):
        reduced_data = PCA(n_components=2).fit_transform(x)
        kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
        X_kmeans = kmeans.fit_transform(reduced_data)

        for point in X_kmeans:
            plt.scatter(point[0], point[1])
        plt.show()

    def plot_gaussian_mixture(self, x):
        mixture = GaussianMixture(n_components=1).fit(x)
        density = mixture.score_samples(x)
        plt.fill(x[:, 0], np.exp(density), fc='#ffaf00', alpha=0.7)
        plt.show()

    def plot_svm(self, x):
        reduced_data = PCA(n_components=2).fit_transform(x)
        X_train, X_test = train_test_split(reduced_data, test_size=0.6, random_state=17)
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(X_train)
        y_predicted_from_training = clf.predict(X_train)
        y_predicted_from_test = clf.predict(X_test)
        training_error = y_predicted_from_training[y_predicted_from_training == -1].size
        testing_error = y_predicted_from_test[y_predicted_from_test == -1].size

        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40
        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='green', s=s)
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([a.collections[0], b1, b2],
                   ["learned frontier", "training observations",
                    "new regular observations"],
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "error train: %d/200 ; errors novel regular: %d/40 ; " % (training_error, testing_error))

        plt.title("Novelty Detection with SVM - 1000 weight samples")

        plt.show()


if __name__ == '__main__':
    BeeClassifier.run()
