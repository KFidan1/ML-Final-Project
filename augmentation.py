
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd
from better_profanity import profanity

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
    cuss_word = []
    f = open('cuss_words.txt','r')
    for line in f:
        cuss_word.append(line.strip())
    
    d1 = pd.DataFrame(data)
    neg = []
    neu = []
    pos = []
    sentence_len = []
    compound = []
    punctuation_count = []
    contain_profanity = []
    num_profanity = []
    for sentence in content:
        #1 if has profanity, 0 if no profanity
        count = 0
        if(profanity.contains_profanity(sentence)):
            contain_profanity.append(1)
            for word in sentence.split():
                if word in cuss_word:
                    count = count + 1
            num_profanity.append(count)
        else:
            contain_profanity.append(0)
            num_profanity.append(count)
     
        #length of sentence
        sentence_len.append(len(sentence))

        #punctuation ratio
        count = 0
        for i in range (0, len(sentence)):
            if sentence[i] in ('!', "," ,"\'" ,";" ,"\"", ".", "-" ,"?"):
                count = count + 1
        punctuation_count.append((count/len(sentence)))

        #vader sentiment
        sentiment_scores(sentence, neg, neu, pos, compound)
    
    d1['sentence_length'] = sentence_len
    d1["neg"] = neg
    d1["neu"] = neu
    d1["pos"] = pos
    d1['compound'] = compound
    d1['punctuation_count'] = punctuation_count
    d1['contain_profanity'] = contain_profanity
    d1['num_profanity'] = num_profanity
    d1.to_csv('data/new_dataset.csv', index=True)