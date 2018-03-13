import numpy as np
import csv

def loadData(filename):    
    #split csv-rows up; array[0] = size 1; array[1] = size 784
    csv_array = np.genfromtxt(filename, delimiter=',', dtype=int)
    #array[row, column]
    samples = csv_array[:,1:]
    labels = csv_array[:,0]
    return samples, labels
def main():
    train_samples, train_labels = loadData("simple_train.csv")
    print("hello")

main()