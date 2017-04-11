import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib
import matplotlib.font_manager
import locale

locale.setlocale(locale.LC_ALL, 'en_US')


class Visualizer(object):
    @staticmethod
    def print_sample_size(hive_id, sample_size):
        formatted_size = locale.format("%d", sample_size, grouping=True)
        print("\nHive #%d with %s messages" % (hive_id, formatted_size))

    @staticmethod
    def output_accuracy_to_console(classifier, training, test):
        training_predictions = classifier.predict(training)
        test_predictions = classifier.predict(test)

        training_error = float(training_predictions[training_predictions == -1].size)
        testing_error = float(test_predictions[test_predictions == -1].size)

        print("\n%.2f%% of data was outliers in training" % (training_error / (float(len(training))) * 100.0))
        print("%.2f%% of data was outliers in testing" % (testing_error / float(len(training)) * 100.0))

    @staticmethod
    def show_contours(classifier, hive_id, training_messages, testing_messages, total_size):
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
            "error train: %d/%d ; errors novel regular: %d/%d ; "
            % (training_error, len(training_messages), testing_error, len(testing_messages)))

        plt.title("Novelty Detection with SVM for Hive #%d - %d weight samples" % (hive_id, (locale.format("%d", total_size, grouping=True))))
        plt.show()

    @staticmethod
    def show_outliers(hive_id, all_messages, inliers):
        plt.plot(all_messages[:, 0], all_messages[:, 1], 'r', label="All messages")
        plt.plot(inliers[:, 0], inliers[:, 1], 'b', label="In-lying Messages")

        plt.legend(loc='upper right')
        plt.title("Novelty Detection with SVM for Hive #%d - %d weight samples" % (hive_id, (len(inliers))))
        plt.show()


    @staticmethod
    def _formatted_time(epoch_time):
        return time.strftime('%Y-%m-%d %I:%M%p', time.localtime(epoch_time))

    @staticmethod
    def _axis_times(epoch_times):
        return np.array([time.strftime('%Y-%m-%d', time.localtime(x)) for x in epoch_times])
