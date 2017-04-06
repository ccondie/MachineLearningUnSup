from matrix import Matrix
from random import randint
from random import uniform
from mpl_toolkits.mplot3d import Axes3D

import operator
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


class Centroid(object):
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'b']
    color_count = 0

    def __init__(self, col_type, col_min, col_max, row=None):
        self.cent_num = Centroid.color_count
        self.color = Centroid.colors[Centroid.color_count]
        Centroid.color_count += 1

        # col_type will store the type ('nominal' or 'continuous') of the indexed col
        self.col_type = col_type
        self.col_min = col_min
        self.col_max = col_max

        # cols will store the "location" of the centroid
        self.location = []
        if row is None:
            for i in range(len(col_type)):
                if col_type[i] == 'continuous':
                    self.location.append(uniform(col_min[i], col_max[i]))
                else:
                    self.location.append(randint(math.floor(col_min[i]), math.ceil(col_max[i])))
        else:
            self.location = row

        # stores the data points connected with this centroid
        self.rows = list()

        print('Initializing Centroid ' + str(self.cent_num) + ': ', end='')
        for el in self.location:
            print(el, end=', ')
        print()

    def add_row(self, instance):
        self.rows.append(instance)

    def clear_instances(self):
        self.rows = []
        pass

    def update_mean(self):
        """
        iterates through the instances contained within this centroid, updating the "location" data in the features list
        :return:
        """
        for col_index in range(len(self.location)):
            new_col_val = 0
            # for each location feature
            if self.col_type[col_index] == 'continuous':
                # column is continuous
                col_sum = 0
                if len(self.rows) == 0:
                    # if during the assignment phase this centroid had no instances, randomly assign it a new location
                    new_col_val = uniform(self.col_min[col_index], self.col_max[col_index])
                else:
                    cont_val_count = 0
                    for row in self.rows:
                        # sum the values for that feature for each of the instances
                        if row[col_index] != float('Inf'):
                            col_sum += row[col_index]
                            cont_val_count += 1
                    if cont_val_count > 0:
                        new_col_val = col_sum / cont_val_count
                    else:
                        new_col_val = float('Inf')
            else:
                # column is nominal
                if len(self.rows) == 0:
                    new_col_val = float('Inf')
                else:
                    vote_map = dict()
                    for row in self.rows:
                        col_val = row[col_index]
                        if col_val == float('Inf'):
                            continue

                        if col_val in vote_map:
                            vote_map[col_val] += 1
                        else:
                            vote_map[col_val] = 1

                    if len(vote_map) == 0:
                        new_col_val = float('Inf')
                    else:
                        new_col_val = self.tie_break(vote_map)

            self.location[col_index] = new_col_val

    def tie_break(self, votes):
        highest_vote = 0
        lowest_feature_id = sys.maxsize

        for id in votes:
            count = votes[id]
            if count > highest_vote:
                highest_vote = count
                lowest_feature_id = id
            elif count == highest_vote:
                if id < lowest_feature_id:
                    highest_vote = count
                    lowest_feature_id = id
        return lowest_feature_id

    def print(self):
        print('CENTROID ' + str(self.cent_num) + ' VALUES: ', end='')
        self.print_location()
        print('CENTROID INSTANCE COUNT: ' + str(len(self.rows)))
        print('SSE of CLUSTER: ' + str(self.sse()))
        print()

    def print_location(self):
        for col_index in range(len(self.location)):
            if self.col_type[col_index] == 'nominal':
                if col_index == len(self.location)-1:
                    print(KMeansLearner.bad_code_practice.attr_value(col_index, self.location[col_index]), end='')
                else:
                    print(KMeansLearner.bad_code_practice.attr_value(col_index, self.location[col_index]), end=',\t')
            else:
                if col_index == len(self.location) - 1:
                    print("{:0.3f}".format(self.location[col_index]), end='')
                else:
                    print("{:0.3f}".format(self.location[col_index]), end=',\t')
        print()

    def sse(self):
        running_sum = 0
        for row in self.rows:
            running_sum += distance(row, self) ** 2
        return running_sum


def distance(row, centroid):
    distance_sum = 0
    for col_index in range(len(row)):
        row_el = row[col_index]
        cent_el = centroid.location[col_index]

        if row_el == float('Inf') or cent_el == float('Inf'):
            distance_sum += 1
            continue

        if centroid.col_type[col_index] == 'nominal':
            if row_el != cent_el:
                distance_sum += 1
            continue

        distance_sum += (cent_el - row_el) ** 2

    return math.sqrt(distance_sum)

# def distance(row, centroid):
#     distance_sum = 0
#     for col_index in range(len(row)):
#         row_el = row[col_index]
#         cent_el = centroid.location[col_index]
#
#         if row_el == float('Inf') or cent_el == float('Inf'):
#             distance_sum += 1
#             continue
#
#         if centroid.col_type[col_index] == 'nominal':
#             if row_el != cent_el:
#                 distance_sum += 1
#             continue
#
#         distance_sum += (cent_el - row_el) ** 2
#
#     return math.sqrt(distance_sum)


class KMeansLearner(object):
    count = 0
    bad_code_practice = None

    def __init__(self):
        self.k = 5
        self.random_start = False
        self.centroids = []
        self.color_count = 0

    def settle_centroids(self, rows):
        """
        :param rows:
        :return:
        """
        settle_count = 5
        carry_over_sse = 100
        KMeansLearner.bad_code_practice = rows

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
        if self.random_start:
            for k in range(self.k):
                self.centroids.append(Centroid(col_type, col_min, col_max))
        else:
            for k in range(self.k):
                self.centroids.append(Centroid(col_type, col_min, col_max, rows.row(k)))

        done = False
        iteration_count = 0

        while not done:
            iteration_count += 1
            done = True

            print('*************************************')
            print('Iteration ' + str(iteration_count))
            print('*************************************')
            print('Making Assignments')
            # assign each row to a centroid
            for row_index in range(rows.rows):
                # print('\t' + str(row_index) + '=', end='')

                row = rows.row(row_index)
                nearest_centroid = 0
                nearest_distance = distance(row, self.centroids[0])
                for centroid_index in range(1, len(self.centroids)):
                    this_distance = distance(row, self.centroids[centroid_index])
                    if this_distance < nearest_distance:
                        nearest_distance = this_distance
                        nearest_centroid = centroid_index
                self.centroids[nearest_centroid].add_row(row)

                print(nearest_centroid, end='')
                # if (row_index + 1) % 10 == 0:
                #     print('\n', end='')
            print()

            total_sse = 0
            for centroid in self.centroids:
                total_sse += centroid.sse()
            print('SSE: ' + "{:0.3f}".format(total_sse))

            # check if the escape parameters have been met
            # if abs(total_sse - carry_over_sse) > 5:
            #     done = False
            done = False

            if not done:
                # if we have yet to reach a stable configuration, realign the cluster and try again
                print('Computing Centroids:')
                # realign the centroids
                for centroid in self.centroids:
                    centroid.update_mean()
                    print('\tCentroid ' + str(centroid.cent_num) + ' = ', end='')
                    centroid.print_location()

                for centroid in self.centroids:
                    centroid.clear_instances()

            time.sleep(5)

        for centroid in self.centroids:
            # output clustering information
            centroid.print()

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
                area.append(np.pi * 2 ** 2)
            # plot the center of the centroid
            x.append(centroid.location[0])
            y.append(centroid.location[1])
            colors.append(centroid.color)
            area.append(np.pi * 6 ** 2)
        plt.scatter(x, y, s=area, c=colors, alpha=1.0)
        plt.pause(0.05)
