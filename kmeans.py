import numpy as np
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



def main():
    k = [3,5,7,12]
    train_samples, train_labels = load_data("simple_train.csv")
    for i in range(len(k)):
        cluster_center = initial_cluster_centers(train_samples,k[i])
        distance = euclidean_distance(train_samples, cluster_center)
        print(distance)


main()