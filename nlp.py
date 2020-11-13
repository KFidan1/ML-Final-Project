# natural language processing method of identifying troll text
# with help from:
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import nltk
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


# load the models here, set to None if it should re-train the model
d2v_loaded = Doc2Vec.load("./models/d2v_1.model")
ml_loaded = joblib.load("./models/ml_1.model")

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

  def train(self, x_train, y_train):
    if self.d2v_model is None:
      tagged_data = self.tag_data(x_train)

      # hyperparameters
      max_epochs = 10
      vec_size = 20
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

      self.d2v_model.save("./models/d2v_1.model")
      print("d2v_model saved")

    if self.ml_model is None:
      # get word embeddings from the trained d2v model
      x_tagged = self.tag_data(x_train)
      x_embedded_array = [self.d2v_model.infer_vector(x[0]) for x in x_tagged]

      # train the ML model from the word embeddings
      # todo try different models
      self.ml_model = LogisticRegression(random_state=0)
      self.ml_model.fit(x_embedded_array, y_train)

      joblib.dump(self.ml_model, "./models/ml_1.model")
      print("ml_model saved")


def makeGraphs(y_true, y_pred):
  # todo make more plots

  matrix = confusion_matrix(y_true, y_pred)

  # Draw a heatmap with the numeric values in each cell
  f, ax = plt.subplots(figsize=(9, 6))
  sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, ax=ax, cmap="RdYlGn", \
              xticklabels=["Not Troll", "Troll"], yticklabels=["Not Troll", "Troll"])
  plt.ylabel('Predicted')
  plt.xlabel('Actual')
  plt.show()

if __name__ == "__main__":
  # read the data, shuffle, and split into train/test
  data = pd.read_csv("./data/dataset.csv", usecols=["content", "annotation"])
  data = data.sample(frac=1, random_state=0).reset_index(drop=True)
  X = data["content"]
  Y = data["annotation"]
  x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=0.2, shuffle=False)

  # will only train the models that aren't provided
  nlp = NLP(d2v_model=d2v_loaded, ml_model=ml_loaded)
  nlp.train(x_train, y_train)

  # get predictions for the test data
  y_pred = nlp.test(x_test)

  # print("pred, actual, tweet")
  # for i in range(len(y_pred)-1):
  #   print(y_pred[i], y_test[i], x_test[i])
  print("accuracy: ", accuracy_score(y_test, y_pred) * 100)
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

  makeGraphs(y_test, y_pred)


