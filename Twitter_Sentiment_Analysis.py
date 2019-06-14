
# coding: utf-8

# In[1]:


# Twitter sentiment analysis 


# In[2]:


import tweepy 
import re 
import pickle

from tweepy import OAuthHandler 

consumer_key  = "your key"
consumer_secret = "your key"
access_token = "your key"
access_secret = "your key"


# In[ ]:


auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_secret)

args = ['facebook']

api = tweepy.API(auth, timeout = 10)

tweet_list = []

query = args[0]
for status in tweepy.Cursor(api.search , q= query+" -filter:retweets", lang = 'en', result_type = "recent").items(100):
    tweet_list.append(status.text)
    


# In[ ]:


tweet_corpus = []

for i in tweet_list:
    review = re.sub(r'^https://t.co/[a-zA-Z0-9]*\s',' ',i)
    review = re.sub(r'\s+https://t.co/[a-zA-Z0-9]*\s',' ',review)
    review = re.sub(r'\s+https://t.co/[a-zA-Z0-9]*$',' ',review)
    review = review.lower()
    review = re.sub(r"what's", "what is", review)
    review = re.sub(r"that's", "that is", review)
    review = re.sub(r"which's", "which is", review)
    review = re.sub(r"she's", "she is", review)
    review = re.sub(r"he's", "he is", review)
    review = re.sub(r"they're", "they are", review)
    review = re.sub(r"who're", "who are", review)
    review = re.sub(r"we're", "we are", review)
    review = re.sub(r'\W', ' ', review)
    review = re.sub(r'\d',' ', review)
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+[a-z]$',' ', review)
    review = re.sub(r'\s+',' ', review)
    tweet_corpus.append(review)
    #sentence = clf.predict(tfidf.transform([review]).toarray())[0]
   # print(review,":",sentence)



# In[ ]:


# Unpickiling the classifier and the tfifd model 

with open('clf.pickle','rb') as f:
    clf = pickle.load(f)
    
with open('TfidfTransformer.pickle','rb') as f:
    tfidf = pickle.load(f)

tweets_analysis = []

for i in tweet_corpus: 
    if clf.predict(tfidf.transform([i]).toarray())[0] == 1:
        tweets_analysis.append('Positive')
    else:
        tweets_analysis.append('Negative')
        


# In[ ]:


import pandas as pd 

df = {'tweets':tweet_corpus, 'tweet_analysis':tweets_analysis}
tweet_sentiment_analysis = pd.DataFrame(df)


tweet_sentiment_analysis

