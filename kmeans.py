import numpy as np
import csv

def loadData(filename, dataSet = []):    
    #split csv-rows up; array[0] = size 1; array[1] = size 784
    csv_array = np.genfromtxt(filename, delimiter=',', dtype=int)
    #number to be at position: [0][x], pixelvalues at: [1][x]
    split = np.split(csv_array,[1],axis=1)
    dataSet.append(split)
    print("hi")
def main():
    trainSet = []
    loadData("simple_train.csv", trainSet)
    test = np.array([trainSet[0][0], trainSet[0][1]])
    test.append(trainSet[0][1])
    print("hello")

main()