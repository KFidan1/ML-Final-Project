#!/usr/bin/env python3
# CS 425 Final Project
# Decision tree implementation using sklearn library to predict troll text


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = pd.read_csv("data/dataset.csv")

#print(data.head())

features = ['sentence_length', 'neg', 'neu', 'pos', 'compound', 'punctuation_count', 'contain_profanity', 'num_profanity']       

X = data[features]
Y = data['annotation']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)  

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print()
print("------CONFUSION MATRIX------")
print(confusion_matrix(y_test, y_pred))
print()
print("------CLASSIFICATION REPORT------")
print(classification_report(y_test, y_pred))
print()
print("ACCURACY: ", accuracy_score(y_test, y_pred))
print()
