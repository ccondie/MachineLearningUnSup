from matrix import Matrix
from random import randint
from random import uniform
from mpl_toolkits.mplot3d import Axes3D

import math
import time
import numpy as np
import matplotlib.pyplot as plt


class Centroid(object):
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'b']
    color_count = 0

    def __init__(self, col_type, col_min, col_max):
        self.last_delta = 0
        self.iterations_without_delta = 0
        self.delta_threshold = 0.1

        self.color = Centroid.colors[Centroid.color_count]
        Centroid.color_count += 1

        # col_type will store the type ('nominal' or 'continuous') of the indexed col
        self.col_type = col_type
        self.col_min = col_min
        self.col_max = col_max

        # cols will store the "location" of the centroid
        self.location = []
        for i in range(len(col_type)):
            if col_type[i] == 'continuous':
                self.location.append(uniform(col_min[i], col_max[i]))
            else:
                self.location.append(randint(math.floor(col_min[i]), math.ceil(col_max)))

        # stores the data points connected with this centroid
        self.rows = list()

    def add_row(self, instance):
        self.rows.append(instance)

    def clear_instances(self):
        self.rows = []

    def update_mean(self):
        """
        iterates through the instances contained within this centroid, updating the "location" data in the features list
        :return:
        """
        for col_index in range(len(self.location)):
            col_sum = 0
            for row in self.rows:
                col_sum += row[col_index]
            if len(self.rows) == 0:
                if self.col_type[col_index] == 'continuous':
                    col_mean = uniform(self.col_min[col_index], self.col_max[col_index])
                else:
                    pass
            else:
                col_mean = col_sum / len(self.rows)
            self.last_delta = abs(self.location[col_index] - col_mean)
            self.location[col_index] = col_mean
        if self.last_delta < self.delta_threshold:
            self.iterations_without_delta += 1
        else:
            self.iterations_without_delta = 0

    def print(self):
        print(self.location)


class KMeansLearner(object):
    count = 0

    def __init__(self):
        self.k = 5
        self.centroids = []
        self.color_count = 0

    def settle_centroids(self, rows):
        """
        :param rows:
        :return:
        """

        settle_count = 5

        plt.ion()

        # calculate the data 'type' of each col
        col_type = []
        col_min = []
        col_max = []
        for col_index in range(rows.cols):
            col_value_count = rows.value_count(col_index)
            if col_value_count == 0:
                col_type.append('continuous')
            else:
                col_type.append('nominal')
            col_min.append(rows.column_min(col_index))
            col_max.append(rows.column_max(col_index))

        # initialize the centroids
        for k in range(self.k):
            self.centroids.append(Centroid(col_type, col_min, col_max))

        plt.show()
        done = False
        while not done:
            done = True
            # assign each row to a centroid
            for row_index in range(rows.rows):
                row = rows.row(row_index)
                nearest_centroid = 0
                nearest_distance = self.distance(row, self.centroids[0])
                for centroid_index in range(1, len(self.centroids)):
                    this_distance = self.distance(row, self.centroids[centroid_index])
                    if this_distance < nearest_distance:
                        nearest_distance = this_distance
                        nearest_centroid = centroid_index
                self.centroids[nearest_centroid].add_row(row)
            self.plot_2d_centroids()

            # realign the centroids
            for centroid in self.centroids:
                centroid.update_mean()
            self.plot_2d_centroids()

            for centroid in self.centroids:
                centroid.clear_instances()

            for centroid in self.centroids:
                if centroid.iterations_without_delta < settle_count:
                    done = False

        plt.show()
        print('Done ... you have 30 seconds')
        time.sleep(30)

    def classify(self):
        """
        :return:
        """

        """
        find the closest centroid
        """
        pass

    def distance(self, row, centroid):
        distance = 0
        for col_index in range(len(row)):
            if centroid.col_type[col_index] == 'continuous':
                distance += (row[col_index] - centroid.location[col_index]) ** 2
            else:
                if centroid.location[col_index] != row[col_index]:
                    distance += 1
        return math.sqrt(distance)

    def plot_2d_centroids(self):
        plt.clf()
        x = []
        y = []
        colors = []
        area = []
        for centroid in self.centroids:
            for row in centroid.rows:
                x.append(row[0])
                y.append(row[1])
                colors.append(centroid.color)
                area.append(np.pi * 3 ** 2)
            # plot the center of the centroid
            x.append(centroid.location[0])
            y.append(centroid.location[1])
            colors.append(centroid.color)
            area.append(np.pi * 10 ** 2)
        plt.scatter(x, y, s=area, c=colors, alpha=1.0)
        plt.pause(0.05)