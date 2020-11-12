# natural language processing method of identifying troll text


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


TRAIN = False


def tag_data(data):
  # todo play around with preprocessing (what is fed into tokenizer)
  # take out stop words, convert to lower, dont change anything at all, etc
  tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_train)]
  return tagged_data


def test(model, data):
  tagged_data = tag_data(data)

  # to find the vector of a document which is not in training data
  test_data = word_tokenize("Get fucking real dude".lower())
  v1 = model.infer_vector(test_data)
  print("V1_infer", v1)

  # to find most similar doc using tags
  similar_doc = model.docvecs.most_similar('1')
  print(similar_doc)

  # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
  print(model.docvecs['1'])
  pass

def train(data):
  tagged_data = tag_data(data)

  # hyperparameters
  max_epochs = 10
  vec_size = 20
  alpha = 0.025

  # dm=1 will preserve the word order, dm=0 is the bow where order is ignroed
  model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
  model.build_vocab(tagged_data)

  # train the doc2vec model
  for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

  model.save("./models/d2v_1.model")
  print("Model saved")

  # todo, loop through each tweet and get the vector representation of it
  # each tweet will have a vector which will have a corresponding label

  return model

if __name__ == "__main__" :
  # read the data, shuffle, and split into train/test
  data = pd.read_csv("./data/dataset.csv", usecols=["content", "annotation"])
  data = data.sample(frac=1, random_state=0).reset_index(drop=True)
  X = data["content"]
  Y = data["annotation"]
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, shuffle=False)

  if TRAIN:
    model = train(data)
  else:
    model = Doc2Vec.load("./models/d2v_1.model")
    print("Model loaded")

  y_pred = test(model, data)

  # todo get accuracy, plots, other metrics


