# -*- coding: utf-8 -*-
import random
class RandomClassifier:
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    def predict(self,x_test):
        prediections = []
        for row in x_test:
            label = random.choice(y_train)
            prediections.append(label)
        return prediections


from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)

my_classifier = RandomClassifier()
my_classifier.fit(x_train,y_train)
prediction = my_classifier.predict(x_test)

print(prediction)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction))