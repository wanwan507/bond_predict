__author__ = 'bingjiegao'

from sklearn import datasets
import numpy as np
from WindPy import *
import math
from datetime import *
w.start()
data1= w.wsd("019002.SH", "weightedrt,accruedinterest,accrueddays,cleanprice,convexity,ytc,ytp,dailycf,calc_accrint", "2016-01-01", "2017-02-23", "yield=0")
data2 = w.wsd("019002.SH", "ytm_b", "2016-01-01", "2017-02-23", "returnType=1")
x = []
y = []
for index in range(0,len(data1.Data[0])):
    today = [];
    for item in range(0, len(data1.Data)):
        if math.isnan(data1.Data[item][index]):
            data1.Data[item][index] = 0
        today.append(data1.Data[item][index])
    x.append(today)
y = data2.Data[0]

x = np.array(x)
y = np.array(y)

data = x
target = y

from sklearn import preprocessing
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data = data_scaler.fit_transform(data)
target = target_scaler.fit_transform(target.reshape(-1,1))

from sklearn.cross_validation import train_test_split
from neupy import environment

environment.reproducible()

#6:2:2 is to small for training set
x_train,x_test,y_train,y_test = train_test_split(data,target,train_size=0.85)

from neupy import algorithms,layers

cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(6),
        layers.Sigmoid(10),
        layers.Sigmoid(1),
    ],
    search_method = 'golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

cgnet.train(x_train,y_train,x_test,y_test,epochs=100)

from neupy import plots
plots.error_plot(cgnet)

from neupy.estimators import rmsle

y_predict = cgnet.predict(x_test).round(1)

error = rmsle(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict))
print error

import matplotlib.pyplot as plt
plt.figure(2)
plt.plot(y_predict,'red')
plt.plot(y_test,'green')
plt.show()
