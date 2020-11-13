# natural language processing method of identifying troll text

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pandas as pd
import nltk
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


TRAIN_D2V = True
TRAIN_ML = None


class NLP:
  def __init__(self, d2v_model=None, ml_model=None):
    self.d2v_model = d2v_model # this is the doc2vec model
    self.ml_model = ml_model   # this is the ml model that is trained on the doc2vec output

  def tag_data(self, data):
    # todo play around with preprocessing (what is fed into tokenizer)
    # take out stop words, convert to lower, dont change anything at all, etc
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    return tagged_data


  def test(self, x_test):
    x_tagged = self.tag_data(x_test)
    x_embedded_array = []
    for x in x_tagged:
      x_embedded_array.append(self.d2v_model.infer_vector(x[0]))

    # get the word embeddings from our trained doc2vec model
    # todo add the preprocessing from above (the .lower() call and todos)
    # x_embedded_array = self.d2v_model.infer_vector(x_test)

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
      x_embedded_array = []
      for x in x_tagged:
        x_embedded_array.append(self.d2v_model.infer_vector(x[0]))

      # print("x_embedded_array[0]", x_embedded_array[0])
      # print("y_train[0]", y_train[0])

      # train the ML model from the word embeddings
      self.ml_model = LogisticRegression()
      self.ml_model.fit(x_embedded_array, y_train)


if __name__ == "__main__":
  # read the data, shuffle, and split into train/test
  data = pd.read_csv("./data/dataset.csv", usecols=["content", "annotation"])
  data = data.sample(frac=1, random_state=0).reset_index(drop=True)
  X = data["content"]
  Y = data["annotation"]
  x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=0.2, shuffle=False)

  # load the doc2vec model if we don't want to train it again
  # nlp = None
  # if TRAIN_D2V:
  #   nlp = NLP()
  #   nlp.train(x_train, y_train)
  # else:
  #   d2v_model = Doc2Vec.load("./models/d2v_1.model")
  #   nlp = NLP(d2v_model=d2v_model)
  #   print("d2v_model loaded")

  d2v_model = Doc2Vec.load("./models/d2v_1.model")
  nlp = NLP(d2v_model=d2v_model)
  nlp.train(x_train, y_train)

  # get predictions for the test data
  y_pred = nlp.test(x_test)

  print("pred, actual, tweet")
  for i in range(len(y_pred)-1):
    print(y_pred[i], y_test[i], x_test[i])
  print("accuracy: ", accuracy_score(y_test, y_pred) * 100)
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

  # todo get accuracy, plots, other metrics


