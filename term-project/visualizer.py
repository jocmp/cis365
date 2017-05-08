import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import dates as mdates
import matplotlib.font_manager
import datetime as dates


class Visualizer(object):
    @staticmethod
    def print_sample_size(hive_id, sample_size):
        print("\nHive #%d with %s messages" % (hive_id, sample_size))

    @staticmethod
    def output_accuracy_to_console(classifier, training, test):
        training_predictions = classifier.predict(training)
        test_predictions = classifier.predict(test)

        training_error = float(training_predictions[training_predictions == -1].size)
        testing_error = float(test_predictions[test_predictions == -1].size)

        print("\n%.2f%% of data were outliers in training" % (training_error / (float(len(training))) * 100.0))
        print("%.2f%% of data were outliers in testing" % (testing_error / float(len(training)) * 100.0))

    @staticmethod
    def show_contours(classifier, hive_id, training_messages, testing_messages, total_size):
        '''
        For original sample graph: http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
        '''
        training_predictions = classifier.predict(training_messages)
        test_predictions = classifier.predict(testing_messages)

        training_error = training_predictions[training_predictions == -1].size
        testing_error = test_predictions[test_predictions == -1].size

        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40
        b1 = plt.scatter(training_messages[:, 0], training_messages[:, 1], c='orange', s=s)
        b2 = plt.scatter(testing_messages[:, 0], testing_messages[:, 1], c='blueviolet', s=s)
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([a.collections[0], b1, b2],
                   ["learned frontier", "training observations",
                    "test observations"],
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "outliers in train: %d/%d ; outliers in test: %d/%d ; "
            % (training_error, len(training_messages), testing_error, len(testing_messages)))

        plt.title("Outlier Detection with SVM for Hive #%d - %d weight samples" % (hive_id, total_size))
        plt.show()

    @staticmethod
    def show_outliers(hive_id, all_messages, inliers):
        plt.plot(Visualizer._convert_dates(all_messages[:, 0]), all_messages[:, 1], 'r',
                 label="All messages")
        plt.plot(Visualizer._convert_dates(inliers[:, 0]), inliers[:, 1], 'b', label="In-lying Messages")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.legend(loc='upper right')
        plt.xlabel('Time')
        plt.ylabel('Weight (kg)')

        plt.show()

    @staticmethod
    def _convert_dates(epoch_times):
        arr = np.array([dates.datetime.fromtimestamp(x) for x in epoch_times])
        return arr
