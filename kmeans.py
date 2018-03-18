import numpy as np
from numpy import inf
import scipy.spatial
import csv

def load_data(filename):    
    #split csv-rows up; array[0] = size 1; array[1] = size 784
    csv_array = np.genfromtxt(filename, delimiter=',', dtype=int)
    #array[row, column]
    samples = csv_array[:,1:]
    return samples

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

def assign_data_to_clusters(train_samples, distance):
    sorted_cluster = []
    clusters = np.unique(distance)
    #assign each element of training set to the cluster with the shortest distance
    for i in range(len(clusters)):
        sorted_cluster.append(np.where(distance==clusters[i])[0])
    return sorted_cluster

def calculate_new_cluster_center(train_samples, sorted_cluster):
    new_center = []
    #calculate new center
    for j in range(len(sorted_cluster)):
        new_center.append(
            np.divide(
                np.sum(train_samples[sorted_cluster[j]], axis=0),
                len(train_samples[sorted_cluster[j]])
            )
        )
    new_center = np.array(new_center)
    return new_center

def calc_dunn_index(train_samples, sorted_cluster):
    max_within_cluster = []
    for i in range(len(sorted_cluster)):
        distance = scipy.spatial.distance.cdist(
            train_samples[sorted_cluster[i]], 
            train_samples[sorted_cluster[i]], 
            metric='euclidean'
        )
        #add max of each cluster to max matrix
        max_within_cluster =  np.append(max_within_cluster,np.amax(distance))
    #get max value of max matrix
    max_within_cluster = np.amax(max_within_cluster)

    min_between_clusters = []
    for i in range(len(sorted_cluster)):
        #starts at an higher turn, to avoid duplicate calculation. e.g. [0,1] == [1,0]
        for j in range(i, len(sorted_cluster)):
            #avoid [0,0], [1,1] etc.
            if(i != j):
                distance = scipy.spatial.distance.cdist(
                    train_samples[sorted_cluster[i]], 
                    train_samples[sorted_cluster[j]], 
                    metric='euclidean'
                )
                min_between_clusters = np.append(min_between_clusters, np.amin(distance))
    min_between_clusters = np.amin(min_between_clusters)
    #calculate dunn index
    index = np.divide(min_between_clusters, max_within_cluster)
    return index



def main():
    #k = 5,7,9,10,12,15
    k = [10]
    train_samples = load_data("train.csv")
    sorted_cluster = []
    for i in range(len(k)):
        cluster_center = initial_cluster_centers(train_samples,k[i])
        for z in range(0, 200):
            distance = euclidean_distance(train_samples, cluster_center)
            sorted_cluster = assign_data_to_clusters(train_samples, distance)
            new_cluster_centers = calculate_new_cluster_center(train_samples, sorted_cluster)
            if(np.array_equal(cluster_center, new_cluster_centers)):
                print("Stopped at run %ith:" % z)
                print("Old and new cluster centers for k: %i are equal!" % k[i])
                break
            cluster_center = new_cluster_centers
        dunn_index = calc_dunn_index(train_samples, sorted_cluster)
        print("k: %i, dunn index: %f" % (k[i],dunn_index))
            
        
main()