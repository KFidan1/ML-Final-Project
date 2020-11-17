
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_validate 
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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

decision_tree = tree.DecisionTreeClassifier()

'''
K-Fold cross validation. we can use this on training data to find best combination of hyperparams to test on untouched test data
'''
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True)

x = x_train.to_numpy()
y = y_train.to_numpy()

confusion_overall = [[0,0], [0,0]]
pred_all = np.array([])
y_all = np.array([])

# PRINT CONFUSION MATRIX FOR EACH K FOLD (uncomment to print confusion and scores for every k fold)
for train_index, val_index in kf.split(x):
    decision_tree.fit(x[train_index], y[train_index])
    pred = decision_tree.predict(x[val_index])
    #print(confusion_matrix(y[val_index], pred))
    #print(classification_report(y[val_index], pred))
    confusion_overall += confusion_matrix(y[val_index], pred)
    pred_all = np.concatenate((pred_all, pred))
    y_all = np.concatenate((y_all, y[val_index]))
    
    

print("CONFUSION MATRIX")
print(confusion_overall)
print(classification_report(y_all, pred_all))

'''
scoring = ('precision', 'recall', 'accuracy', 'f1')
cv_results = cross_validate(decision_tree, X, Y, cv=kf, scoring=scoring, return_train_score=False)
print(cv_results)
print("-------AVERAGE RESULTS-------")

for score in scoring:
    print(score, round(cv_results['test_' + score].mean(), 4))
'''


#details about the tree
r = export_text(decision_tree, feature_names = ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity'])
f = open("tree_details.txt", "w")
f.write(r)
f.close()

print("Tree Depth = ", decision_tree.get_depth())
print()
print("Tree Leaf = ", decision_tree.get_n_leaves())
print()

print("______________________________________NEXT TREE DEPTH = 3______________________________________")
print()

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)

confusion_overall = [[0,0], [0,0]]
pred_all = np.array([])
y_all = np.array([])

# PRINT CONFUSION MATRIX FOR EACH K FOLD (uncomment to print confusion and scores for every k fold)
for train_index, val_index in kf.split(x):
    decision_tree.fit(x[train_index], y[train_index])
    pred = decision_tree.predict(x[val_index])
    #print(confusion_matrix(y[val_index], pred))
    #print(classification_report(y[val_index], pred))
    confusion_overall += confusion_matrix(y[val_index], pred)
    pred_all = np.concatenate((pred_all, pred))
    y_all = np.concatenate((y_all, y[val_index]))

print("CONFUSION MATRIX")
print(confusion_overall)
print(classification_report(y_all, pred_all))

#details about the tree
r = export_text(decision_tree, feature_names = ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity'])
f = open("tree_details.txt", "w")
f.write(r)
f.close()

print("Tree Depth = ", decision_tree.get_depth())
print()
print("Tree Leaf = ", decision_tree.get_n_leaves())
print()

fn= ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity']
cn=['Troll', 'Not Troll']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(decision_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True)
plt.show()
