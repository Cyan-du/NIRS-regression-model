'''最终版本 自动编码机和grnn结合 完成预测'''
#!/usr/bin/env python
# encoding: utf-8
import os
import pandas as pd
from keras.layers import Dense, Input
import numpy as np
np.random.seed(1337)
from keras.models import Model
import numpy as np
from neupy import algorithms, estimators, environment


def read_ds(fn):
    ds = pd.read_csv(fn,header=None)
    X = ds.iloc[0:10778, 0:1557].values
    y = ds.iloc[0:10778, 1557:1558].values
    X_valid = ds.iloc[10778:, 0:1557].values
    y_valid = ds.iloc[10778:, 1557:1558].values
    return X,X_valid,y,y_valid

def train_encoder(X_combine,wn):

    '''初始版本'''
    encoding_dim = 30
    input_img = Input(shape=(1557,))

    encoded = Dense(200, activation='relu')(input_img)
    encoded = Dense(100, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    decoded = Dense(100, activation='relu')(encoder_output)
    decoded = Dense(200, activation='relu')(decoded) #(decoded)
    decoded = Dense(1557, activation='tanh')(decoded)

    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoder_output)
    
    if(not(os.path.exists(wn))):
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # training
        autoencoder.fit(X_combine, X_combine,
                        epochs=30,
                        batch_size=128,
                        shuffle=True)
    
        autoencoder.save_weights('ae_weights.h5')
    else:
        autoencoder.load_weights('ae_weights.h5')

    return encoder.predict(X_combine)

def scorer(network, X, y):
    result = network.predict(X)
    return estimators.rmsle(result, y)

if __name__ == "__main__":
    '''判断之前的降维部分，是否已经训练过'''
    wn = "C:\\Users\\Administrator\\Desktop\\stat-learn-master\\RBFNet\\ae_weights.h5"
    fn = 'C:\\Users\\Administrator\\Desktop\\totaldata.csv'
    X,X_valid,y,y_valid = read_ds(fn)
    X_combine = np.concatenate((X,X_valid),axis=0)

    # #################归一化###################
    # X_combine = preprocessing.scale(X_combine)

    ############ auto-encoder ###############
    encoded_X_combine = train_encoder(X_combine,wn)
    print(encoded_X_combine.shape)

    X1 = encoded_X_combine[0:10778,:]
    X2 = encoded_X_combine[10778:,:]

    print(X1.shape)
    print(X2.shape)

    # rbf = RBFNet(k=40)
    # rbf.fit(X1,y)
    # prediction = rbf.predict(X2)
    
    ##################GRNN#######################
    nw = algorithms.GRNN(std=0.015, verbose=False)
    nw.train(X1, y)
    error = scorer(nw,X2,y_valid)

    print("GRNN RMSLE = {:.3f}\n".format(error))
    prediction = nw.predict(X2)

    real = y_valid.flatten()
    pred = prediction.flatten()

    for i in range(len(y_valid)):
        print('real      :', real[i])
        print('prediction:', pred[i])
        print('*' * 30)

    diff = abs(real - pred)
    right = diff[diff<= 0.001]
    acc = len(right)/len(y_valid)

    print("准确率为：")
    print(acc)