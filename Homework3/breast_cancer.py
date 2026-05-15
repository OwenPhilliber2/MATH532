from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import random
from sparse_SVM import sparse_SVM

cd = os.getcwd()

mat_data = loadmat(os.path.join(cd, 'Homework3', 'data', 'wisconsin_breast_cancer.mat'))

data = np.array(mat_data['X'])
labels = mat_data['y']
labels = np.array(labels * 2 - 1)

m = data.shape[0]
n = data.shape[1]

random.seed(1)
# Initialize indeces
ind = list(range(m))
# Shuffle the indeces
shuffle(ind)

# Split the indices into a training set, 70%, validations set, 15%, and test set, 15%
tr_ind = ind[:int(m * .70)]
val_ind = ind[int(m * .70):int(m * .85)]
test_ind = ind[int(m * .85):]

# Segmentting the data
tr_data = data[tr_ind, :]
tr_labels = labels[tr_ind]

val_data = data[val_ind, :]
val_labels = labels[val_ind]

test_data = data[test_ind, :]
test_labels = labels[test_ind]

# Computing the weights and accuracies
C_vals =  np.logspace(-5, 5, num=11, base=10.0)
accuracies = []

for i in C_vals:
    w_temp, b_temp = sparse_SVM(X = tr_data, y = tr_labels, C = i)
    w_temp = w_temp.reshape(-1,1).T

    pred_labels = np.sign(w_temp @ val_data.T + b_temp)

    acc = np.mean(pred_labels == val_labels.T)
    accuracies.append(acc)

    num_predictors = np.count_nonzero(w_temp)
    
    print(f"C Value: {i:.2e}, Accuracy: {acc:.2f}, Number of Non Zero Predictors: {num_predictors}")

C = C_vals[np.argmax(accuracies)]
print(f"Optimal C value: {C}")

# Results with C = 1
w, b = sparse_SVM(X = tr_data, y = tr_labels, C = C)

w_mag = np.abs(w)
plt.plot(np.sort(w_mag)[::-1]+1e-12)
plt.title("Weights in the Model Ordered by their Absolute Magnitude")
plt.xlabel("Index of the Weights Ordered by their Absolute Magnitude")
plt.ylabel("Log Scale of the Absolute Value of W_i")
plt.yscale("log")
plt.show()

test_p_labels = np.sign(w @ test_data.T + b)

print(f"Testing accuracy: {np.mean(test_p_labels == test_labels.T)*100:.2f}%")

# Creating the confusion matrix
TP = np.count_nonzero((test_p_labels == 1) & (test_labels.T == 1))
TN = np.count_nonzero((test_p_labels == -1) & (test_labels.T == -1))
FN = np.count_nonzero((test_p_labels == -1) & (test_labels.T == 1))
FP = np.count_nonzero((test_p_labels == 1) & (test_labels.T == -1))

print(f"Confusion Matrix: \n {TP} {FN} \n {FP} {TN}")
