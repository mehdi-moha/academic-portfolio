# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:55:57 2023

@author: Mehdi Mohammadi
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

IRIS = datasets.load_iris()

X = IRIS.data
Y_Data = IRIS.target

pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X)

kpca = KernelPCA(n_components = 3)
X_kpca = kpca.fit_transform(X)

sc = StandardScaler()
X = sc.fit_transform(X)
X_pca = sc.fit_transform(X_pca)
X_kpca = sc.fit_transform(X_kpca)

X_Data = np.column_stack((X, X_pca, X_kpca))

Xtr, Xte, Ytr, Yte = train_test_split(X_Data, Y_Data, train_size = 0.5, random_state = 1)

NN = MLPClassifier(solver = 'lbfgs', alpha = 1e-3, hidden_layer_sizes = (7, 3))
NN.fit(Xtr, Ytr);
Ypr = NN.predict(Xte)

train_error_rate = 1 - NN.score(Xtr, Ytr)
test_error_rate = 1 - NN.score(Xte, Yte)

print('train Error Rate: ', train_error_rate)
print('test Error Rate: ', test_error_rate)