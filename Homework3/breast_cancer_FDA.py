from FDA import FDA
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from random import shuffle


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

X_tr = tr_data[tr_labels.flatten() == 1, :].T
Y_tr = tr_data[tr_labels.flatten() == -1, :].T

X_val = val_data[val_labels.flatten() == 1, :].T
Y_val = val_data[val_labels.flatten() == -1, :].T

X_test = test_data[test_labels.flatten() == 1, :].T
Y_test = test_data[test_labels.flatten() == -1, :].T

reduced_x, reduced_y, w = FDA(X_tr, Y_tr)

val_x = w.T @ X_val
val_y = w.T @ Y_val

test_x = w.T @ X_test
test_y = w.T @ Y_test

threshold_values = np.linspace(
    min(reduced_x.min(), reduced_y.min()),
    max(reduced_x.max(), reduced_y.max()),
    num=50
)

best_threshold = (min(reduced_x.min(), reduced_y.min()), 0)

for i in threshold_values:
    x_correct = np.sum(val_x >= i)
    y_correct = np.sum(val_y < i)

    tot_corr = x_correct + y_correct

    acc = tot_corr / (Y_val.shape[1] + X_val.shape[1])

    print(f"Threshold value: {i:.4f}, Accuracy of validation set: {acc:.2f}")

    if acc > best_threshold[1]:
        best_threshold = (i, acc)

test_acc_x = np.sum(test_x >= best_threshold[0])
test_acc_y = np.sum(test_y < best_threshold[0])

tot_corr_test = test_acc_x + test_acc_y

test_acc = tot_corr_test / (Y_test.shape[1] + X_test.shape[1])

plt.hist(test_x, alpha=0.5, label='X', bins=30)
plt.hist(test_y, alpha=0.5, label='Y', bins=30)
plt.vlines(best_threshold[0], 0, 4, colors = "black", label = 'threshold')
plt.legend(loc='upper right')
plt.title("Test Reduced Calues")
plt.show()

print(f"Best validation accuracy: {best_threshold[1]:.2f} from a threshold value of: {best_threshold[0]:.5f}. Testing accuracy: {test_acc:.4f}")



    

