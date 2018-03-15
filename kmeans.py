import numpy as np
from numpy import inf
from numpy import nan
import scipy.spatial
import csv

def load_data(filename):    
    #split csv-rows up; array[0] = size 1; array[1] = size 784
    csv_array = np.genfromtxt(filename, delimiter=',', dtype=int)
    #array[row, column]
    samples = csv_array[:,1:]
    labels = csv_array[:,0]
    return samples, labels

def initial_cluster_centers(train_samples, k):
    cluster_center = np.empty([k,784], dtype=int)
    for i in range(k):
        cluster_center[i] = train_samples[i]
    return cluster_center

def euclidean_distance(train_samples, cluster_center):
    distance = scipy.spatial.distance.cdist(train_samples, cluster_center, metric='euclidean')
    best_distance = np.argpartition(distance,1)
    best_distance = best_distance[:,0]
    return best_distance

def calculate_new_cluster_center(train_samples, distance):
    #only works for k=3, needs to change
    clusters = np.unique(distance)
    cluster_1 = np.where(distance==clusters[0])[0]
    cluster_2 = np.where(distance==clusters[1])[0]
    cluster_3 = np.where(distance==clusters[2])[0]

    test = np.divide(
        np.sum(train_samples[cluster_1], axis=0),
        np.absolute(train_samples[cluster_1])
        )
    test[(test == inf) | (np.isnan(test))] = 0

    return 2




def main():
    #k = 5,7,9,10,12,15
    k = [3]
    train_samples, train_labels = load_data("simple_train.csv")
    for i in range(len(k)):
        cluster_center = initial_cluster_centers(train_samples,k[i])
        distance = euclidean_distance(train_samples, cluster_center)
        new_cluster_centers = calculate_new_cluster_center(train_samples, distance)
        print(distance)

main()