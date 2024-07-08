from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()
features = iris.data
x= features[:,3:]
targets = iris.target 
y = (targets ==2).astype(int)
# print(x)
# print(y)
clf = LogisticRegression()
clf.fit(x,y)
pred = clf.predict([[2.6]])
if (pred==1):
    print("yes virginica")
else:
    print("nope, not virginica")
# print(pred)
xn = np.linspace(0,3,490).reshape(-1,1)
# print(xn)
xp = clf.predict_proba(xn)
# print(xp)
plt.plot(xn,xp[:,1],"g-")
plt.show()
