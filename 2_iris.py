# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree 
import numpy as np
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])
for i in range(len(iris.target)):
        print("Example:%d lable %s feature %s"%(i,iris.target[i],iris.data[i])) 
        
test_index = [0,50,100]

# training data
train_target = np.delete(iris.target,test_index)
train_data = np.delete(iris.data,test_index,axis=0)

# testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

print(test_target)
print(clf.predict(test_data))

# viz code
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 