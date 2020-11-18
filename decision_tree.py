
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV 
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

tuned_parameters = [
    {
        'max_depth': [2, 4, 6, 8, 10],
        'min_impurity_decrease': [1e-1, 1e-2, 1e-3, 1e-4]
    }
]

scoring = ('precision', 'recall','accuracy')

for score in scoring:
        clf = GridSearchCV(decision_tree, tuned_parameters, scoring=score, refit = "accuracy",return_train_score=True)
        clf.fit(x_train, y_train)

        print("best parameters are: ")
        print(clf.best_params_)
        print("Grid scores on development set:")
        print()
        
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()
        


decision_tree = tree.DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.0001)
decision_tree.fit(x_train, y_train)
pred = decision_tree.predict(x_test)

print("-----RESULTS OF BEST HYPERPARAMETERS------")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


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
