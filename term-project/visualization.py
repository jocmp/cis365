import seaborn as sns
import spc
import numpy as np
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self, n):
        self.init = "init"

    def plotControlChart(self, x):
        cc = spc.Spc(x, spc.CHART_X_MR_X)
        cc.get_chart()
        plt.show()

    def plotClusters(self, data, clusters, centroids, ax=None, holdon=False):
        sns.set(style="white")

        self.data = data

        if ax is None:
            _, ax = plt.subplots()

        for i, index in enumerate(clusters):
            point = np.array(data[index]).T
            ax.scatter(*point, c=sns.color_palette("hls", self.K + 1)[i])

        for point in centroids:
            ax.scatter(*point, marker='x', linewidths=10)

        if not holdon:
            plt.show()
