import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/dataset.csv')

features = ['sentence_length', 'compound','neg' , 'neu' , 'pos', 'punctuation_count', 'contain_profanity', 'num_profanity']
bins = {
    'sentence_length' : 296,
    'compound' : 10,
    'neg' : 10,
    'neu' : 10,
    'pos' : 10,
    'punctuation_count' : 10,
    'contain_profanity' : 10,
    'num_profanity' : 10,
}
xlimit = {
    'sentence_length' : 200,
    'compound' : 1,
    'neg' : 1,
    'neu' : 1,
    'pos' : 1,
    'punctuation_count' : 0.5,
    'contain_profanity' : 1,
    'num_profanity' : 3.5,
}
X = data[features]

for feat in X:
    # feature vs count
    plt.hist(data[feat], bins = bins[feat], rwidth=0.9)
    plt.title("{} vs. # of tweets".format(feat))
    plt.xlabel("{}".format(feat))
    plt.ylabel("# of Tweets")
    plt.xlim(0, xlimit[feat])
    plt.savefig('data_count/count/{}_count.png'.format(feat), format='png')
    plt.clf()

    # feature vs annotation
    plt.scatter(data[feat], data["annotation"], s=2)
    plt.title("{} vs. Troll Tweet".format(feat))
    plt.xlabel("{}".format(feat))
    plt.ylabel("Troll Label")
    axes = plt.gca()
    axes.set_ylim([-0.7, 1.7])
    plt.yticks([0, 1])
    plt.savefig('data_count/output/{}_vs_annotation.png'.format(feat), format='png')
    plt.clf()



