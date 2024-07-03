from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
# print(iris.DESCR)
features = iris.data
labels = iris.target 
# print(features[1],labels[1])
clf  = KNeighborsClassifier()
clf.fit(features,labels)
pred = clf.predict([[1.2,3.4,5.6,2.2]])
if pred==[0]:
    print("Setosa")
elif pred==[1]:
    print("versicolour")
else:
    print("virginicia")
# print(pred)
