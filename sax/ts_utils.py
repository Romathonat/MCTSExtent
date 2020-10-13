import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def readECG():
    original_data = pd.read_csv('../data/ecg.csv')
    data = original_data.values.flatten()

    discretisation_nb = 10

    data = data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=discretisation_nb, random_state=0).fit(data)

    output_data = []

    # data from 0 to 207 are positives (anomaly)
    # print(data[2])
    # print(kmeans.labels_[2])
    # print(kmeans.cluster_centers_[kmeans.labels_[2]])

    for i, line in enumerate(original_data.values):
        if i < 208:
            new_seq = ['+']
        else:
            new_seq = ['-']
        for number in line:
            new_seq.append(set(kmeans.predict([[number]])))
        output_data.append(new_seq)

    return output_data, kmeans


def print_pattern_discretization(results, kmeans):
    for result in results:
        pattern = result[1]
        output_pattern = 'Quality: {}, Pattern:'.format(result[0])

        for elt in pattern:
            output_pattern += '{} '.format(kmeans.cluster_centers_[next(iter(elt))])
        print(output_pattern)

def read_gun_point():
    original_data = pd.read_csv('../data/gunpoint.csv')
    data = original_data.iloc[:, 1:].values.flatten()

    discretisation_nb = 10

    data = data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=discretisation_nb, random_state=0).fit(data)

    output_data = []

    for i, line in enumerate(original_data.values):
        new_seq = [str(int(line[0]))]
        for number in line[1:]:
            new_seq.append(set(kmeans.predict([[number]])))
        output_data.append(new_seq)

    return output_data, kmeans


# with open('../data/ecg_20.csv', mode='w') as file:
#     file_writer = csv.writer(file, delimiter=',')
#
#     for line in readECG():
#         file_writer.writerow(line)
