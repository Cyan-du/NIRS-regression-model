'''最终版本 pca和grnn结合 完成预测'''
#!/usr/bin/env python
# encoding: utf-8
import numpy as np
np.random.seed(1337)
from neupy import algorithms,estimators
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

def scorer(network, X, y):
    result = network.predict(X)
    return estimators.rmsle(result, y)

ds = pd.read_csv('totaldata.csv', header=None)
X = ds.iloc[0:10778, 0:1557].values
y = ds.iloc[0:10778, 1557:1558].values
X_valid = ds.iloc[10778:, 0:1557].values
y_valid = ds.iloc[10778:, 1557:1558].values

pca = PCA(n_components=30)
'''尝试了kernel的pca，没有传统pca跑得快，且应用在光谱数据上效果一般'''
# kpca = KernelPCA(kernel="rbf",fit_inverse_transform=True,gamma=10,n_components=10)

X_combine = np.concatenate((X,X_valid),axis=0)

# #################标准化###################
# X_combine = preprocessing.scale(X_combine)

min_max_scaler = preprocessing.MinMaxScaler()
X_combine = min_max_scaler.fit_transform(X_combine)

#######grnn###########
reduced_X_combine = pca.fit_transform(X_combine)
print(reduced_X_combine.shape)

X1 = reduced_X_combine[0:10778,:]
X2 = reduced_X_combine[10778:,:]

print(X1.shape)
print(X2.shape)
print(pca.explained_variance_ratio_)

nw = algorithms.GRNN(std=0.013, verbose=False)
nw.train(X1, y)
prediction = nw.predict(X2)
error = scorer(nw,X2,y_valid)
print("GRNN RMSLE = {:.3f}\n".format(error))

real = y_valid.flatten()
pred = prediction.flatten()

rmse = estimators.rmse(pred, real)
print("the rmse is :",rmse)

for i in range(len(y_valid)):
    print('real      :', real[i])
    print('prediction:', pred[i])
    print('*' * 30)

diff = abs(real - pred)
right = diff[diff<= 0.001]
acc = len(right)/len(y_valid)


print("准确率为：")
print(acc)
