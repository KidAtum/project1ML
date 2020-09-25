from turtle import st

import numpy as np
import math
import pandas
import scipy
from sklearn.datasets import load_iris
from sklearn import datasets, linear_model, metrics
import pandas as pd
import random

import collections
from scipy.stats import entropy



print("Printing Binning Data Set...")
# load iris data set
dataset = load_iris()
a = dataset.data
b = np.zeros(150)

# take 1st column among 4 column of data set
for i in range(150):
    b[i] = a[i, 1]

b = np.sort(b)  # sort the array

# create bins
bin1 = np.zeros((30, 5))
bin2 = np.zeros((30, 5))
bin3 = np.zeros((30, 5))

# Bin mean
for i in range(0, 150, 5):
    k = int(i / 5)
    mean = (b[i] + b[i + 1] + b[i + 2] + b[i + 3] + b[i + 4]) / 5
    for j in range(5):
        bin1[k, j] = mean
print("Bin Mean: \n", bin1)

# Bin boundaries
for i in range(0, 150, 5):
    k = int(i / 5)
    for j in range(5):
        if (b[i + j] - b[i]) < (b[i + 4] - b[i + j]):
            bin2[k, j] = b[i]
        else:
            bin2[k, j] = b[i + 4]
print("Bin Boundaries: \n", bin2)

# Bin median
for i in range(0, 150, 5):
    k = int(i / 5)
    for j in range(5):
        bin3[k, j] = b[i + 2]
print("Bin Median: \n", bin3)


#SAMPLING
aList = [2.2,2.2,2.2,2.2, 2.2,
 2.3, 2.3 ,2.3, 2.3, 2.3,
 2.5, 2.5, 2.5, 2.5, 2.5,
 2.5, 2.5, 2.5, 2.5, 2.5,
 2.6, 2.6, 2.6, 2.6, 2.6,
 2.7, 2.7, 2.7, 2.7, 2.7,
 2.7, 2.7, 2.7, 2.7, 2.7,
 2.8, 2.8, 2.8, 2.8, 2.8,
 2.8, 2.8 ,2.8, 2.8, 2.8,
 2.9 ,2.9 ,2.9 ,2.9 ,2.9,
 2.9 ,2.9, 2.9, 2.9, 2.9,
 3.  ,3. , 3.  ,3. , 3. ,
 3.  ,3. , 3.,  3.,  3. ,
 3.  ,3. , 3. , 3. , 3. ,
 3. , 3.,  3.,  3.,  3. ,
 3. , 3. , 3. , 3. , 3. ,
 3. , 3.,  3.,  3.,  3. ,
 3.1 ,3.1 ,3.1 ,3.1 ,3.1,
 3.1 ,3.1, 3.1, 3.1, 3.1,
 3.2 ,3.2 ,3.2 ,3.2 ,3.2,
 3.2 ,3.2, 3.2, 3.2, 3.2,
 3.3, 3.3, 3.3, 3.3, 3.3,
 3.3 ,3.3, 3.3, 3.3, 3.3,
 3.4 ,3.4 ,3.4 ,3.4 ,3.4,
 3.4 ,3.4, 3.4, 3.4, 3.4,
 3.5 ,3.5 ,3.5 ,3.5 ,3.5,
 3.6 ,3.6 ,3.6, 3.6, 3.6,
 3.7 ,3.7, 3.7 ,3.7 ,3.7,
 3.8 ,3.8, 3.8, 3.8, 3.8,
 4.1 ,4.1 ,4.1 ,4.1 ,4.1]
print ("choosing 20 random items from Bin Median")
sampled_list = random.sample(aList, 20)
print(sampled_list)


#ENTROPY
print("Doing Entropy Calculations on Bin Median...")

def entropy_cal(array):

    total_entropy = 0

    for i in array:
        total_entropy += -i * math.log(2, i)

    return total_entropy

def main():

    probabilities = [2.2,2.2,2.2,2.2, 2.2,
 2.3, 2.3 ,2.3, 2.3, 2.3,
 2.5, 2.5, 2.5, 2.5, 2.5,
 2.5, 2.5, 2.5, 2.5, 2.5,
 2.6, 2.6, 2.6, 2.6, 2.6,
 2.7, 2.7, 2.7, 2.7, 2.7,
 2.7, 2.7, 2.7, 2.7, 2.7,
 2.8, 2.8, 2.8, 2.8, 2.8,
 2.8, 2.8 ,2.8, 2.8, 2.8,
 2.9 ,2.9 ,2.9 ,2.9 ,2.9,
 2.9 ,2.9, 2.9, 2.9, 2.9,
 3.  ,3. , 3.  ,3. , 3. ,
 3.  ,3. , 3.,  3.,  3. ,
 3.  ,3. , 3. , 3. , 3. ,
 3. , 3.,  3.,  3.,  3. ,
 3. , 3. , 3. , 3. , 3. ,
 3. , 3.,  3.,  3.,  3. ,
 3.1 ,3.1 ,3.1 ,3.1 ,3.1,
 3.1 ,3.1, 3.1, 3.1, 3.1,
 3.2 ,3.2 ,3.2 ,3.2 ,3.2,
 3.2 ,3.2, 3.2, 3.2, 3.2,
 3.3, 3.3, 3.3, 3.3, 3.3,
 3.3 ,3.3, 3.3, 3.3, 3.3,
 3.4 ,3.4 ,3.4 ,3.4 ,3.4,
 3.4 ,3.4, 3.4, 3.4, 3.4,
 3.5 ,3.5 ,3.5 ,3.5 ,3.5,
 3.6 ,3.6 ,3.6, 3.6, 3.6,
 3.7 ,3.7, 3.7 ,3.7 ,3.7,
 3.8 ,3.8, 3.8, 3.8, 3.8,
 4.1 ,4.1 ,4.1 ,4.1 ,4.1]
    entropy = entropy_cal(probabilities)

    print(entropy)

if __name__=="__main__":
    main()

#BAYES THEORY
print("Doing Bayes Theory...")

# Could not do it in python. I couldnt figure it out (the code).
# But I did it in the text file I submitted. So maybe partial credit? :-) if not, i understand
