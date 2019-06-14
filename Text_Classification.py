
# coding: utf-8

# # Text Classification

# In[11]:


# Importing the libraries

import numpy as np 
from nltk.corpus import stopwords
import re
import pickle 
import nltk
from sklearn.datasets import load_files
import pandas as pd


# In[12]:


# Importing the dataset
dataset = load_files("txt_sentoken/")

X, y = dataset.data , dataset.target


# In[13]:


# storing the X, y as a pickle file to reduce the size of the file 

with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)


# In[14]:


# Unpickeling the dataset 
del X
del y

with open('X.pickle','rb') as f:
    X = pickle.load(f)

with open('y.pickle','rb') as f:
    y = pickle.load(f)
    
    


# In[15]:


# Text Preprocessing 

# Creating the corpus

corpus = []

for i in range(0,len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+',' ', review)
    corpus.append(review)


# In[24]:


'''from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer(max_features= 2000,min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()'''


# In[59]:


'''from sklearn.feature_extraction.text import TfidfTransformer 

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()'''


# In[16]:


# We can do the above two seperate steps as Count vectroizer followed by Tfidf transformer or in a single step like below
from sklearn.feature_extraction.text import TfidfVectorizer 


vectorizer = TfidfVectorizer(max_features= 2000,min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


# In[17]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 0)


# In[18]:


# Logistic Regression 

from sklearn.linear_model import LogisticRegression 
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score 

prediction = accuracy_score(y_test, y_pred)

prediction 


# In[19]:


# Picikiling the model for further use 

with open('clf.pickle','wb') as f:
    pickle.dump(clf,f)
    
with open('TfidfTransformer.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    


# In[36]:


# Unpickiling the classifier and the tfifd model 

with open('clf.pickle','rb') as f:
    clf = pickle.load(f)
    
with open('TfidfTransformer.pickle','rb') as f:
    tfidf = pickle.load(f)
    
    
sample = ['I love this world']

sample = tfidf.transform(sample).toarray()

if clf.predict(sample)[0] == 1:
    print('It is a positive review')
else:
    print('It is a negative review')


# # Twitter sentiment analysis 

# In[50]:


import tweepy 
import re 
import pickle

from tweepy import OAuthHandler 

consumer_key  = "your key"
consumer_secret = "your key"
access_token = "your key"
access_secret = "your key"

# In[51]:


auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_secret)

args = ['facebook']

api = tweepy.API(auth, timeout = 10)

tweet_list = []

query = args[0]
for status in tweepy.Cursor(api.search , q= query+" -filter:retweets", lang = 'en', result_type = "recent").items(100):
    tweet_list.append(status.text)
    


# In[71]:


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



# In[72]:


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
        


# In[88]:


import pandas as pd 

df = {'tweets':tweet_corpus, 'tweet_analysis':tweets_analysis}
tweet_sentiment_analysis = pd.DataFrame(df)


tweet_sentiment_analysis

