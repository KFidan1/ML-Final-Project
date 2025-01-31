
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
import seaborn as sns
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def makeGraphsPerRun(y_true, y_pred, name):
  matrix = confusion_matrix(y_true, y_pred)

  # Draw a heatmap with the numeric values in each cell
  f, ax = plt.subplots(figsize=(6, 6))
  sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, ax=ax, cmap="RdYlGn", \
              xticklabels=["Not Troll", "Troll"], yticklabels=["Not Troll", "Troll"])
  plt.ylabel('Predicted')
  plt.xlabel('Actual')

  if(name != None):
    plt.title(name)
  plt.savefig("./plots/decision_tree/" + name + ".png")


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
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True)

#Running with no hyperparameters
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
pred = decision_tree.predict(x_test)

print("-----RESULTS OF NO HYPERPARAMETERS------")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
makeGraphsPerRun(y_test, pred, "Best result for decision tree with no hyperparameters")
print("Tree Depth = ", decision_tree.get_depth())
print()
print("Tree Leaf = ", decision_tree.get_n_leaves())
print()


#these were used for the course grid search
tuned_parameters = [
    {
        'max_depth': [2, 4, 6, 10, 20, 30, 40, 80],
        'min_impurity_decrease': [1e-1, 1e-2, 1e-3, 1e-4]
    }
]

#these are used for the fine grid search
fine_tuned_parameters = [
    {
        'max_depth': [30, 40, 45, 50],
        'min_impurity_decrease': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6] 
    }
]

scoring = ('precision', 'recall','accuracy')

for score in scoring:
        clf = GridSearchCV(decision_tree, fine_tuned_parameters, scoring=score, refit = "accuracy",return_train_score=True)
        clf.fit(x_train, y_train)

        print("best parameters are: ")
        print(clf.best_params_)
        if(score == "accuracy"):
                best = clf.best_params_
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
        y_true, y_pred = y_val, clf.predict(x_val)
        print(classification_report(y_true, y_pred))
        print()
        makeGraphsPerRun(y_val, y_pred, score)
        

print("-----BEST HYPERPARAMETERS---------")
print(best)
decision_tree = tree.DecisionTreeClassifier(max_depth=best['max_depth'], min_impurity_decrease=best['min_impurity_decrease'])
decision_tree.fit(x_train, y_train)
pred = decision_tree.predict(x_test)

print("-----RESULTS OF BEST HYPERPARAMETERS------")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


makeGraphsPerRun(y_test, pred, "Best hyperparameters for decision tree")

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
#plt.show()
