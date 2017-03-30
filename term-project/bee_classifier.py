import numpy as np
import pandas as pd
#from sklearn.cluster import KMeans

TRAINING_DATA = './just_weight.csv'
TRAINING_DATA2 = './all_attrs.csv'

class beeClassifier:
    def __init__(self, K=5, max_iters=100, init='random'):
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.init = init

    def run():
        weight = pd.read_csv(TRAINING_DATA)
        bee_data = pd.read_csv(TRAINING_DATA2)

        print('Test: %.3f' % 10.5)

    def get_centroid(self, cluster):
        return [np.mean(np.take(self.X[:, i], cluster)) for i in range(self.n_features)]

    def dist_from_centers(self):
        print (np.array([min([euclidean_distance(x, c) for c in self.centroids]) for x in self.X]))

if __name__ == '__main__':
    beeClassifier.run()
