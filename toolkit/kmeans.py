from matrix import Matrix
from random import randint
from random import uniform
from mpl_toolkits.mplot3d import Axes3D

import operator
import math
import sys
import time
import copy
import numpy as np


# import matplotlib.pyplot as plt


def tie_break(votes):
    """
    Used to calculate and return a nominal_enum from a map of nominal_enum:vote_counts
    :param votes:
    :return:
    """
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


def distance(row, cent_row):
    distance_sum = 0
    for col_index in range(len(row)):
        row_el = row[col_index]
        cent_el = cent_row[col_index]

        if row_el == float('Inf') or cent_el == float('Inf'):
            # if either of the values is missing ... add a distance of 1
            distance_sum += 1
            continue

        if KMeansLearner.col_type[col_index] == 'nominal':
            # if the value is nominal and not equal ... add a distance of 1
            if row_el != cent_el:
                distance_sum += 1
            continue

        # if the value is continuous ... default distance
        distance_sum += math.pow(cent_el - row_el, 2)

    return math.sqrt(distance_sum)


class Centroid(object):
    centroid_count = 0

    def __init__(self, row=None):
        self.cent_num = Centroid.centroid_count
        Centroid.centroid_count += 1

        # cols will store the "location" of the centroid
        self.location = []
        if row is None:
            for i in range(len(KMeansLearner.instance_data.row(0))):
                if KMeansLearner.col_type[i] == 'continuous':
                    self.location.append(uniform(KMeansLearner.col_min[i], KMeansLearner.col_max[i]))
                else:
                    self.location.append(
                        randint(math.floor(KMeansLearner.col_min[i]), math.ceil(KMeansLearner.col_max[i])))
        else:
            self.location = copy.deepcopy(row)

        # stores the data points connected with this centroid
        self.instances = list()

    def add_row(self, instance):
        self.instances.append(instance)

    def clear_instances(self):
        self.instances = []

    def update_mean(self):
        """
        iterates through the instances contained within this centroid, updating the "location" data in the features list
        :return:
        """
        for col_index in range(len(self.location)):
            new_col_val = 0
            # for each location feature
            if KMeansLearner.col_type[col_index] == 'continuous':
                col_sum = 0
                if len(self.instances) == 0:
                    # if during the assignment phase this centroid had no instances, randomly assign it a new location
                    new_col_val = uniform(KMeansLearner.col_min[col_index], KMeansLearner.col_max[col_index])
                else:
                    val_count = 0
                    for instance in self.instances:
                        # sum the values for that feature for each of the instances
                        if instance[col_index] != float('Inf'):
                            col_sum += instance[col_index]
                            val_count += 1
                    if val_count > 0:
                        new_col_val = col_sum / val_count
                    else:
                        new_col_val = float('Inf')
            else:
                # column is nominal
                if len(self.instances) == 0:
                    new_col_val = float('Inf')
                else:
                    vote_map = dict()
                    for instance in self.instances:
                        col_val = instance[col_index]
                        if col_val == float('Inf'):
                            continue
                        else:
                            if col_val in vote_map:
                                vote_map[col_val] += 1
                            else:
                                vote_map[col_val] = 1

                    if len(vote_map) == 0:
                        new_col_val = float('Inf')
                    else:
                        new_col_val = tie_break(vote_map)
            # Commit the new column value into the centroid location
            self.location[col_index] = new_col_val

    def print_location(self):
        for col_index in range(len(self.location)):
            if self.location[col_index] == float('Inf'):
                if col_index == len(self.location) - 1:
                    print('?', end='')
                else:
                    print('?', end=',\t')
                continue
            if KMeansLearner.col_type[col_index] == 'nominal':
                if col_index == len(self.location) - 1:
                    print(KMeansLearner.instance_data.attr_value(col_index, self.location[col_index]), end='')
                else:
                    print(KMeansLearner.instance_data.attr_value(col_index, self.location[col_index]), end=',\t')
            else:
                if col_index == len(self.location) - 1:
                    print("{:0.3f}".format(self.location[col_index]), end='')
                else:
                    print("{:0.3f}".format(self.location[col_index]), end=',\t')
        print()

    def sse(self):
        running_sum = 0
        for instance in self.instances:
            running_sum += distance(instance, self.location) ** 2
        return running_sum


class KMeansLearner(object):
    count = 0
    instance_data = None
    col_type = []
    col_max = []
    col_min = []

    def __init__(self):
        self.k = 5
        self.random_start = False
        self.centroids = []

    def settle_centroids(self, instances):
        carry_over_sse = sys.maxsize
        KMeansLearner.instance_data = instances

        # calculate the data 'type' of each col
        for col_index in range(instances.cols):
            col_value_count = instances.value_count(col_index)
            if col_value_count == 0:
                KMeansLearner.col_type.append('continuous')
            else:
                KMeansLearner.col_type.append('nominal')
            KMeansLearner.col_min.append(instances.column_min(col_index))
            KMeansLearner.col_max.append(instances.column_max(col_index))

        done = False
        iteration_count = 0

        while not done:
            iteration_count += 1
            done = True

            print('*************************************')
            print('Iteration ' + str(iteration_count))
            print('*************************************')

            # ************************************************************************************************
            # Compute Centroids
            # ************************************************************************************************
            print('Computing Centroids:')
            if iteration_count == 1:
                # initialize the centroids
                if self.random_start:
                    for k in range(self.k):
                        self.centroids.append(Centroid())
                else:
                    for k in range(self.k):
                        self.centroids.append(Centroid(instances.row(k)))
            else:
                # update the centroids
                for centroid in self.centroids:
                    centroid.update_mean()

            # print the new centroid locations
            for centroid in self.centroids:
                print('\tCentroid ' + str(centroid.cent_num) + ' = ', end='')
                centroid.print_location()

            # ************************************************************************************************
            # Assign to Centroids
            # ************************************************************************************************
            print('Making Assignments')
            # clear the existing centroid instances
            for centroid in self.centroids:
                centroid.clear_instances()
            for instance_index in range(instances.rows):
                print('\t' + str(instance_index) + '=', end='')
                instance = instances.row(instance_index)
                nearest_centroid = 0
                nearest_distance = distance(instance, self.centroids[0].location)
                for centroid_index in range(1, len(self.centroids)):
                    this_distance = distance(instance, self.centroids[centroid_index].location)
                    if this_distance < nearest_distance:
                        nearest_distance = this_distance
                        nearest_centroid = centroid_index
                self.centroids[nearest_centroid].add_row(instance)

                print(nearest_centroid, end='')
                if (instance_index + 1) % 10 == 0:
                    print('\n', end='')
            print()

            # ************************************************************************************************
            # calculate the SSE
            # ************************************************************************************************
            total_sse = 0
            for centroid in self.centroids:
                total_sse += centroid.sse()
            print('SSE: ' + "{:0.3f}".format(total_sse))
            print()

            if total_sse != carry_over_sse:
                done = False
                carry_over_sse = total_sse