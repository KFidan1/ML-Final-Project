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

classifier_list = ["DecisionTreeClassifier", "GaussianNB", "LogisticRegression", "SVC", "KNeighborsClassifier"]

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

  #TODO, update to support parameters
  def generate_models(self):
    print("Generate models")
    
    #Previous code
    if(self.single_model != None):
      try:
        loaded = joblib.load(f"./models/{self.num_vectors} {self.single_model}.model")
        self.tup.append((loaded, True))
      except:
        self.tup.append((LogisticRegression(random_state=0), False))
    else:
      for model_name in classifier_list:
        try:
          loaded = joblib.load(f"./models/{self.num_vectors} {model_name}.model")
          print("Loaded")
          self.tup.append((loaded, True))
        except:
          self.tup.append((LogisticRegression(random_state=0), False))
          if(model_name == "DecisionTreeClassifier"):
            self.tup.append((DecisionTreeClassifier(), False))
          elif(model_name == "GaussianNB"):
            self.tup.append((GaussianNB(), False))
          elif(model_name == "LogisticRegression"):
            self.tup.append((LogisticRegression(random_state=0), False))
          elif(model_name == "SVC"):
            self.tup.append((SVC(kernel='poly', C=10, degree=3), False))
          elif(model_name == "KNeighborsClassifier"):
            self.tup.append((KNeighborsClassifier(), False))

  def test_models(self):
    print("Testing")

    self.generate_models()

    data = pd.read_csv("./data/classificationdata.csv")

    num_data = data[data["vectors"] == 20]
    listdata = data[data["classifier"] == 'KNeighborsClassifier']
    svcdata = data[data["classifier"] == 'SVC']

    #I just need to duplicate this for other classifiers, which is easy
    svc_obj = []
    for svc in svcdata.itertuples():
      try:
        loaded = joblib.load(f"./models/{self.num_vectors} {svc[2]} {svc[3]} {svc[4]} {svc[5]}.model")
        svc_obj.append((Classifier("SVC", loaded, svc), True))
      except:
        if(svc[3] == "poly"):
          svc_class = SVC(kernel=svc[3], C=svc[4], degree=svc[5])
        elif(svc[3] == "rbf"):
          svc_class = SVC(kernel=svc[3], C=svc[4], gamma=svc[5])
        elif(svc[3] == "linear"):
          svc_class = SVC(kernel=svc[3], C=svc[4])
        svc_obj.append((Classifier("SVC", svc_class, svc), False))

    #This is a terrible way to do this, I want something that works though right now.
    #like this is finna about to take so much memory
    #Also, I think there might be a bug where the first thing gets ran a lot
    for model in svc_obj:
      try:
        d2v = Doc2Vec.load(f"./models/d2v_{self.num_vectors}.model")
      except:
        d2v = None

      nlp = NLP(d2v, model[0].classifier)

      print(model[0].output)

      if(model[1] == False):
        nlp.train(x_train, y_train ,self.num_vectors)
        svc = model[0].output
        joblib.dump(nlp.ml_model, f"./models/{self.num_vectors} {svc[2]} {svc[3]} {svc[4]} {svc[5]}.model")
        
      #Evil optimization hack
      self.d2v_model = nlp.d2v_model

      y_pred = nlp.test(x_test)

      # print("pred, actual, tweet")
      # for i in range(len(y_pred)-1):
      #   print(y_pred[i], y_test[i], x_test[i])
      print("accuracy: ", accuracy_score(y_test, y_pred) * 100)
      print(confusion_matrix(y_test, y_pred))
      print(classification_report(y_test, y_pred))
      makeGraphs(y_test, y_pred, str(model[0].output))

    #For normal models, etc.
    for model in self.tup:
      nlp = NLP(self.d2v_model, model[0])

      if(model[1] == False):
        nlp.train(x_train, y_train ,self.num_vectors)
        joblib.dump(nlp.ml_model, f"./models/{self.num_vectors} {nlp.ml_model.__class__.__name__}.model")

      #Evil optimization hack
      self.d2v_model = nlp.d2v_model

      y_pred = nlp.test(x_test)

      # print("pred, actual, tweet")
      # for i in range(len(y_pred)-1):
      #   print(y_pred[i], y_test[i], x_test[i])
      print("accuracy: ", accuracy_score(y_test, y_pred) * 100)
      print(confusion_matrix(y_test, y_pred))
      print(classification_report(y_test, y_pred))
      makeGraphs(y_test, y_pred, model[0].__class__.__name__)

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

      # hyperparameters
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

def makeGraphs(y_true, y_pred, name = None):
  # todo make more plots
  matrix = confusion_matrix(y_true, y_pred)

  # Draw a heatmap with the numeric values in each cell
  f, ax = plt.subplots(figsize=(9, 6))
  sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, ax=ax, cmap="RdYlGn", \
              xticklabels=["Not Troll", "Troll"], yticklabels=["Not Troll", "Troll"])
  plt.ylabel('Predicted')
  plt.xlabel('Actual')
  
  if(name != None):
    plt.title(name)

  plt.show()

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

  nlpTester = NLPTester(num_vectors=20, single_model = inputname)
  nlpTester.test_models()
