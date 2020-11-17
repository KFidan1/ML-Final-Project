# natural language processing method of identifying troll text
# with help from:
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import nltk
import sys

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

class Classifier:
  def __init__(self, name, classifier, output):
    self.name = name
    self.classifier = classifier
    self.output = output

class NLPTester:
  def __init__(self, num_vectors=20, single_model = None):
    self.d2v_model = None
    self.num_vectors = num_vectors
    self.single_model = single_model
    self.ml_models = []
    self.tup = []
    self.accuracies = []

  def getModels(self):
    return [str(model[0].output)[(str(model[0].output)).find("'"):(str(model[0].output)).rfind("'")+1] for model in self.ml_models]

  def getAccuracies(self):
    return self.accuracies

  #TODO, update to support parameters
  def generate_models(self):
    print("Generate models")

    models = []
    #This actually doesn't work yet
    if(self.single_model != None):
      try:
        print(self.single_model)
        loaded = joblib.load(self.single_model)
        return models.append(Classifier("SVC", loaded, []), True)
      except:
        print("File doesn't exist, exiting program")
        sys.exit()

    data = pd.read_csv("./data/classificationdata.csv")
    gaussdata = data[data["classifier"] == 'GaussianNB']
    knndata = data[data["classifier"] == 'KNeighborsClassifier']
    svcdata = data[data["classifier"] == 'SVC']
    logdata = data[data["classifier"] == 'LogisticRegression']

    #I just need to duplicate this for other classifiers, which is easy
    for svc in svcdata.itertuples():
      try:
        loaded = joblib.load(f"./models/{svc[1]} {svc[2]} {svc[3]} {svc[4]} {svc[5]}.model")
        models.append((Classifier("SVC", loaded, svc), True))
      except:
        if(svc[3] == "poly"):
          svc_class = SVC(kernel=svc[3], C=svc[4], degree=svc[5])
        elif(svc[3] == "rbf"):
          svc_class = SVC(kernel=svc[3], C=svc[4], gamma=svc[5])
        elif(svc[3] == "linear"):
          svc_class = SVC(kernel=svc[3], C=svc[4])
        models.append((Classifier("SVC", svc_class, svc), False))

    for knn in knndata.itertuples():
      try:
        loaded = joblib.load(f"./models/{knn[1]} {knn[2]} {knn[3]}.model")
        models.append((Classifier("KNeighborsClassifier", loaded, knn), True))
      except:
        knn_class = KNeighborsClassifier(n_neighbors=int(knn[3]))
        models.append((Classifier("KNeighborsClassifier", knn_class, knn), False))

    for gaussian in gaussdata.itertuples():
      try:
        loaded = joblib.load(f"./models/{gaussian[1]} {gaussian[2]} {gaussian[3]}.model")
        models.append((Classifier("GaussianNB", loaded, gaussian), True))
      except:
        gaussian_class = GaussianNB(var_smoothing = float(gaussian[3]))
        models.append((Classifier("GaussianNB", gaussian_class, gaussian), False))

    for logistic in logdata.itertuples():
      try:
        loaded = joblib.load(f"./models/{logistic[1]} {logistic[2]} {logistic[3]}.model")
        models.append((Classifier("LogisticRegression", loaded, logistic), True))
      except:
        logistic_class = LogisticRegression(C = int(logistic[3]))
        models.append((Classifier("LogisticRegression", logistic_class, logistic), False))

    self.ml_models = models
    return models
  
  def test_models(self):
    print("Testing")

    svc_obj = self.generate_models()

    #This is a terrible way to do this, I want something that works though right now.
    #like this is finna about to take so much memory
    #Also, I think there might be a bug where the first thing gets ran a lot
    for model in svc_obj:
      svc = model[0].output
      try:
        d2v = Doc2Vec.load(f"./models/d2v_{svc[1]}.model")
      except:
        d2v = None

      nlp = NLP(d2v, model[0].classifier)

      print(model[0].output)
      if(model[1] == False):
        nlp.train(x_train, y_train, svc[1])
        if(model[0].name == "SVC"):
          joblib.dump(nlp.ml_model, f"./models/{svc[1]} {svc[2]} {svc[3]} {svc[4]} {svc[5]}.model")
        else:
          joblib.dump(nlp.ml_model, f"./models/{svc[1]} {svc[2]} {svc[3]}.model")
      else:
        print("Loading existing model")

      y_pred = nlp.test(x_test)

      # print("pred, actual, tweet")
      # for i in range(len(y_pred)-1):
      #   print(y_pred[i], y_test[i], x_test[i])


      print("accuracy: ", accuracy_score(y_test, y_pred) * 100)
      print(confusion_matrix(y_test, y_pred))
      print(classification_report(y_test, y_pred))
      self.accuracies.append(accuracy_score(y_test, y_pred) * 100)
      makeGraphsPerRun(y_test, y_pred, str(model[0].output))


class NLP:
  def __init__(self, d2v_model=None, ml_model=None):
    self.d2v_model = d2v_model # this is the doc2vec model to construct the word binding arrays
    self.ml_model = ml_model   # this is the ml model that is trained on the doc2vec arrays

  def preprocess(self, tweet):
    # todo take out stop words, dont change anything at all, etc
    return tweet.lower()

  def tag_data(self, data):
    tagged_data = [TaggedDocument(words=word_tokenize(self.preprocess(tweet)), tags=[str(i)]) for i, tweet in enumerate(data)]
    return tagged_data

  def test(self, x_test):
    # get the word embeddings from our trained doc2vec model
    x_tagged = self.tag_data(x_test)
    x_embedded_array = [self.d2v_model.infer_vector(x[0]) for x in x_tagged]

    # use the trained ML model to predict from each embedding
    y_pred = self.ml_model.predict(x_embedded_array)
    return y_pred

  def train(self, x_train, y_train, numv):
    #TODO: Remove duplicate code
    if self.d2v_model is None:
      print(f"Creating doc2vec for # vectors = {numv}")
      tagged_data = self.tag_data(x_train)

      # todo try different hyperparams for d2v
      max_epochs = 10
      vec_size = numv
      alpha = 0.025

      # dm=1 will preserve the word order, dm=0 is the bow where order is ignroed
      self.d2v_model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
      self.d2v_model.build_vocab(tagged_data)

      # train the doc2vec model
      for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        self.d2v_model.train(tagged_data, total_examples=self.d2v_model.corpus_count, epochs=self.d2v_model.epochs)

        # decrease the learning rate
        self.d2v_model.alpha -= 0.0002
        # fix the learning rate, no decay
        self.d2v_model.min_alpha = self.d2v_model.alpha

      self.d2v_model.save(f"./models/d2v_{vec_size}.model")
      print("d2v_model saved")

    #Fix this to support both fit and unfit, DOES NOT WORK %100 yet
    if self.ml_model is not None:
      # get word embeddings from the trained d2v model
      x_tagged = self.tag_data(x_train)
      x_embedded_array = [self.d2v_model.infer_vector(x[0]) for x in x_tagged]

      # train the ML model from the word embeddings
      #print(self.ml_model.get_params())
      self.ml_model.fit(x_embedded_array, y_train)
      
      #joblib.dump(self.ml_model, f"./models/{numv} {self.ml_model.__class__.__name__}.model")
      print("ml_model saved")

# this makes the plots for each run of the models
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

  plt.savefig("./plots/" + name + ".png")

# this makes the plots at the end of all the runs. should only run once
def makeFinalGraphs(model_names, accuracies):
  # bar graphs of different models vs accuracy
  f, ax = plt.subplots(figsize=(6, 6))
  colors = sns.color_palette("husl", len(model_names))
  xticks = [i for i in range(len(model_names))]

  for i in range(len(model_names)):
    plt.bar(xticks[i], accuracies[i], width=0.9, color=colors[i], label=model_names[i])

  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand",
             borderaxespad=0.)  # https://stackoverflow.com/questions/44413020
  plt.xticks(xticks, "")
  plt.ylim(64, 76)
  ax.set(ylabel="Accuracy", xlabel="Model")
  plt.savefig("./plots/accuracies.png", bbox_inches='tight')


if __name__ == "__main__":
  # read the data, shuffle, and split into train/test
  data = pd.read_csv("./data/dataset.csv", usecols=["content", "annotation"])
  data = data.sample(frac=1, random_state=0).reset_index(drop=True)
  X = data["content"]
  Y = data["annotation"]
  x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=0.2, shuffle=False)

  if(len(sys.argv) > 1):
    inputname = sys.argv[1]
  else:
    inputname = None

  nlpTester = NLPTester(num_vectors=20, single_model=inputname)
  nlpTester.test_models()

  model_names = nlpTester.getModels()
  accuracies = nlpTester.getAccuracies()
  print(model_names)
  print(accuracies)
  makeFinalGraphs(model_names, accuracies)
