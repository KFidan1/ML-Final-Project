
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd 

def sentiment_scores(sentence): 
    sid_obj = SentimentIntensityAnalyzer() 
  
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    
# Driver code 
if __name__ == "__main__" : 
    data = pd.read_csv('data/dataset.csv')
    content = data['content']
    annotation = data['annotation']
    for sentence in content:
        sentiment_scores(sentence)