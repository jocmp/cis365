import matplotlib
import pandas as pd
import time
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import scale


class HiveClassifier(object):
    def __init__(self, hive_id):
        self.hive_id = hive_id
        self.data_frame = self.hive_data_frame()

    def hive_data_frame(self):
        filename = './data/hive_' + str(self.hive_id) + '_messages_with_weight.csv'
        return pd.read_csv(filename)

    def run(self):
        '''
        http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
        :return: void - visual printout
        '''
        X = self.data_frame.get_values()
        unscaled_training, unscaled_test = train_test_split(X, train_size=0.60, random_state=17)

        unscaled_training.sort(axis=0)
        unscaled_test.sort(axis=0)

        X_train = scale(unscaled_training)
        X_test = scale(unscaled_test)

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(X_train)

        self.show_contours(X_train, X_test, clf)
        self.show_outliers(clf, scale(X), X)

    def show_contours(self, training, testing, classifier):
        y_predicted_from_training = classifier.predict(training)
        y_predicted_from_test = classifier.predict(testing)

        training_error = y_predicted_from_training[y_predicted_from_training == -1].size
        testing_error = y_predicted_from_test[y_predicted_from_test == -1].size

        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40
        b1 = plt.scatter(training[:, 0], training[:, 1], c='orange', s=s)
        b2 = plt.scatter(testing[:, 0], testing[:, 1], c='blueviolet', s=s)
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([a.collections[0], b1, b2],
                   ["learned frontier", "training observations",
                    "test observations"],
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "error train: %d/%d ; errors novel regular: %d/%d ; "
            % (training_error, len(training) / 2, testing_error, len(testing) / 2))

        plt.title(
            "Novelty Detection with SVM for Hive #%d - %d weight samples" % (
                self.hive_id, (len(self.data_frame.get_values()))))
        plt.show()

    def show_outliers(self, clf, X_test, test_data):
        test_predictions = clf.predict(X_test)

        outlier_indices = []
        for i in range(len(test_predictions)):
            if test_predictions[i] == -1:
                print("outlier ", test_data[i])
                outlier_indices.append(i)

        normalized_data = np.array([test_data[index] for index in range(len(test_data))
                                    if index not in outlier_indices])

        plt.clf()
        plt.plot(test_data[:, 0], test_data[:, 1], 'r', label="Outliers")
        plt.plot(normalized_data[:, 0], normalized_data[:, 1], 'b', label="Inliers")
        plt.legend(loc='upper right')
        plt.title("Novelty Detection with SVM for Hive #%d - %d weight samples" % (self.hive_id, (len(X_test))))
        plt.show()

    def plot_hive_weight(self):
        plt.plot(self.data_frame.get_values()[:, 1])
        times = self.data_frame.get_values()[:, 0]
        start_time = self.formatted_time(times[0])
        end_time = self.formatted_time(times[len(times) - 1])
        plt.title("Hive #%d Weight\n%s to %s" % (self.hive_id, start_time, end_time))
        plt.xlabel("Time")
        plt.ylabel("Weight (kg)")

        plt.show()

    def formatted_time(self, epoch_time):
        return time.strftime('%Y-%m-%d %I:%M%p', time.localtime(epoch_time))


if __name__ == '__main__':
    # Possible hives: 11, 49, 137
    classifier = HiveClassifier(hive_id=137)
    classifier.run()
