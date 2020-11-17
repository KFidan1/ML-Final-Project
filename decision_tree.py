import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.tree import export_text
import graphviz
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
data = pd.read_csv('data/dataset.csv', ',')

features = ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity']

X = data[features]
X['sentence_length'] = pd.to_numeric(X['sentence_length'], downcast="float")
X['compound'] = pd.to_numeric(X['compound'], downcast="float")
X['punctuation_count'] = pd.to_numeric(X['punctuation_count'], downcast="float")
X['contain_profanity'] = pd.to_numeric(X['contain_profanity'], downcast="float")
X['num_profanity'] = pd.to_numeric(X['num_profanity'], downcast="float")

Y = data['annotation']
Y = pd.to_numeric(Y, downcast="float")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree = decision_tree.fit(x_train, y_train)
y_predit = decision_tree.predict(x_test)
y_test = y_test.to_numpy()

print("Tree accuracy \t\t-> ", str(accuracy_score(y_test, y_predit.round())))

#details about the tree
r = export_text(decision_tree, feature_names = ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity'])
f = open("tree_details.txt", "w")
f.write(r)
f.close()

print("Tree Depth = ", decision_tree.get_depth())
print()
print("Tree Leaf = ", decision_tree.get_n_leaves())
print()

print(classification_report(y_test, y_predit.round()))

fn= ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity']
cn=['Troll', 'Not Troll']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(decision_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True)
plt.show()
