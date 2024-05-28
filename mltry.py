import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
dia = datasets.load_diabetes()
#['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
print(dia.keys())
print(dia.data)
#diax = dia.data[:, np.newaxis, 2]
diax = dia.data
diax_train = diax[:-30]
diax_test = diax[0:20]
diay_train = dia.target[:-30]
diay_test = dia.target[0:20]
model = linear_model.LinearRegression()
model.fit(diax_train,diay_train)
diay_predict = model.predict(diax_test)
print("Mean squared values : ", mean_squared_error(diay_test, diay_predict))
print("weights", model.coef_)
print("intercept: ", model.intercept_)
#plt.scatter(diax_test,diay_test)
#plt.plot(diax_test,diay_predict)
#plt.show()
