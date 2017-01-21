# -*- coding: utf-8 -*-
from sklearn import tree
# feature [Weight,Texture 0:Bumpy 1:Smooth]
features = [[140,1],[130,1],[150,0],[170,0]]
# lable 0:Apple 1:Orange
lables = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,lables)
print("The result is %d" % clf.predict([[150,0]]))