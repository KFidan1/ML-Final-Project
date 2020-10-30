
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd 

def sentiment_scores(sentence, neg, neu, pos, compound): 
    sid_obj = SentimentIntensityAnalyzer() 
    sentiment_dict = sid_obj.polarity_scores(sentence)
    #print("Overall sentiment dictionary is : ", sentiment_dict) 
    neg.append(sentiment_dict["neg"])
    neu.append(sentiment_dict["neu"])
    pos.append(sentiment_dict["pos"])
    compound.append(sentiment_dict["compound"])
    
# Driver code 
if __name__ == "__main__" : 
    data = pd.read_csv('data/dataset.csv')
    content = data['content']
    annotation = data['annotation']
    d1 = pd.DataFrame(data)
    neg = []
    neu = []
    pos = []
    sentence_len = []
    compound = []
    for sentence in content:
        sentence_len.append(len(sentence))
        sentiment_scores(sentence, neg, neu, pos, compound)
    
    d1['sentence_length'] = sentence_len
    d1["neg"] = neg
    d1["neu"] = neu
    d1["pos"] = pos
    d1['compound'] = compound
    d1.to_csv('data/new_dataset.csv', index=True)