#!/usr/bin/env python3
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from visualizer import Visualizer


class HiveClassifier(object):
    def __init__(self, hive_id):
        self.hive_id = hive_id
        self.data_frame = self.hive_data_frame()
        self.total_size = len(self.data_frame.get_values())

    def hive_data_frame(self):
        filename = './hive_' + str(self.hive_id) + '_messages_with_weight.csv'
        return pd.read_csv(filename)

    def run_with_visualizations(self):
        """
        Unused in main, but will call graph visualizations
        """
        self.run(visualizations=True)

    def run(self, visualizations=False):
        """
        Scales sample data, runs classifier and optionally shows graphs
        """
        X = self.data_frame.get_values()
        Visualizer.print_sample_size(self.hive_id, self.total_size)

        unscaled_training, unscaled_test = train_test_split(X, train_size=0.6, random_state=31)

        # Sort samples based on time
        unscaled_training.sort(axis=0)
        unscaled_test.sort(axis=0)

        scaler = StandardScaler().fit(unscaled_training)
        X_train = self.find_within_std_deviation(scaler.transform(unscaled_training))
        X_test = scaler.transform(unscaled_test)

        clf = svm.OneClassSVM(nu=0.2, gamma=(1 / self.total_size))
        clf.fit(X_train)
        clf.fit(X_test)

        Visualizer.output_accuracy_to_console(clf, X_train, X_test)

        if visualizations:
            inliers = self.get_inliers(clf, scaler.transform(X), X)
            Visualizer.show_outliers(self.hive_id, X, inliers)
            # Visualizer.show_contours(clf, self.hive_id, X_train, X_test, self.total_size)

    @staticmethod
    def get_inliers(svm, scaled_messages, all_messages):
        """
        Predicts using classifier and filters outliers
        :param svm: classifier 
        :param scaled_messages: scaled for prediction
        :return: inlier messages using weight and time
        """
        predictions = svm.predict(scaled_messages)

        outlier_indices = [i for i in range(len(predictions)) if predictions[i] == -1]
        return np.array([all_messages[index] for index in range(len(all_messages)) if index not in outlier_indices])

    @staticmethod
    def find_within_std_deviation(scaled_training):
        """
        :param scaled_training: Scaled with standard deviation 
        :return: normal inliers for training 
        """
        weight = 1
        return np.array([message for message in scaled_training if 1 > message[weight] > -1])


if __name__ == '__main__':
    selected_hives = [11, 49, 137]  # Hive IDs associated with CSV files
    for selected_hive in selected_hives:
        classifier = HiveClassifier(hive_id=selected_hive)
        classifier.run_with_visualizations()
