import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
ax = np.array([[1],[2],[3]])
ax_train = ax
ax_test = ax
ay_train = np.array([3,2,4])
ay_test =np.array([3,2,4])
model = linear_model.LinearRegression()
model.fit(ax_train,ay_train)
ay_predict = model.predict(ax_test)
print("Mean squared values : ", mean_squared_error(ay_test, ay_predict))
print("weights", model.coef_)
print("intercept: ", model.intercept_)
plt.scatter(ax_test,ay_test)
plt.plot(ax_test,ay_predict)
plt.show()
