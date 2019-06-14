
# coding: utf-8

# In[1]:


import re 

words = ["Hello how *ar3e && you  ",
        "I    love     you",
         "1234 done with #you"]

for i in range(len(words)):
    words[i] = re.sub(r"\W"," ",words[i])
    words[i] = re.sub(r"\d"," ",words[i])
    words[i] = re.sub(r"\s+"," ",words[i])
    words[i] = re.sub(r"^\s","",words[i])
    words[i] = re.sub(r"\s$","",words[i])
    words[i] = re.sub(r"\s+[a-z]\s+"," ",words[i])
    
print(words)    


# In[1]:


import nltk
import re 
import heapq
import numpy as np


# In[70]:


paragraph = """Native Americans lived in the Americas for thousands of years.
 English people in 1607 went to the place now called Jamestown, Virginia. 
 Other European settlers went to the colonies, mostly from England and later Great Britain. France, Spain, and the Netherlands also colonized North America. 
 In 1775, a war between the thirteen colonies and Britain began when the colonists were upset over changes in British policies.
 On July 4, 1776, rebel leaders made the United States Declaration of Independence. 
 They won the Revolutionary War and started a new country. 
 They signed the constitution in 1787 and the Bill of Rights in 1791. 
 George Washington, who had led the war, became its first president. 
 During the 19th century, the United States gained much more land in the West and began to become industrialized. 
 In 1861, several states in the South left the United States to start a new country called the Confederate States of America.
 This caused the American Civil War. After the war, Immigration resumed. 
 Some Americans became very rich in this Gilded Age and 
 the country developed one of the largest economies in the world."""


# In[2]:


# Importing the required Libraries
import nltk
from nltk import PorterStemmer 
from nltk import WordNetLemmatizer 


# In[23]:


# converting the paragraph in to sentencens in a list 
sentence = nltk.sent_tokenize(paragraph)


# In[24]:


# Importing the stemmer function 
stemmer = PorterStemmer()

# In the loop we are converting the sentence in to seperate words in a new list and implementing the stemmer 
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    new_words = [stemmer.stem(j) for j in words]
    sentence[i] = ' '.join(new_words)

  


# In[25]:


# Importing the Lemmatizer function 
sentence = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer() 

for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    new_words = [lemmatizer.lemmatize(j) for j in words]
    sentence[i] = ' '.join(new_words)



# In[26]:


nltk.download('stopwords')


# In[34]:


# Stopwords 

# Importing the stopwords from nltk corpus
from nltk.corpus import stopwords

# Sentence Tokenization 
sentence = nltk.sent_tokenize(paragraph)

# Converting the sentence in to words using word tokenization and using stop words for english to remove unwanted words
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    new_words = [ word for word in words if word not in stopwords.words('english') ]
    sentence[i] = ' '.join(new_words)


# In[59]:


# Parts of Speech 

# converting the sentence in to words
words = nltk.word_tokenize(paragraph)

tagged_words = nltk.pos_tag(words)

new_tags = []


for i in tagged_words:
    new_tags.append(i[0] + "_" + i[1])
    
tagged_paragraph = ' '.join(new_tags)


# In[64]:


# Named Entities Recognition 

para = 'The Taj Mahal was build by Emporer Shah Jahan'
# tokenizing the words

words = nltk.word_tokenize(para)

# tagging the words 
tagged_words = nltk.pos_tag(words)

# Named entity will create a tree of words in to seperate category
namedEntity = nltk.ne_chunk(tagged_words)

# use Draw() function to see the tree 
namedEntity.draw()


# In[71]:


# building bag of words model 

sentence = nltk.sent_tokenize(paragraph)

for i in range(len(sentence)):
    sentence[i]= sentence[i].lower()
    sentence[i] = re.sub('\W',' ',sentence[i])
    sentence[i] = re.sub('\s+',' ',sentence[i])
    


# In[74]:


# Creating the count of words

word2vec = {}

for i in sentence:
    words = nltk.word_tokenize(i)
    for j in words:
        if j not in word2vec.keys():
            word2vec[j] = 1
        else:
            word2vec[j] += 1
            


# In[92]:


# Considering only the top 50 word counts 
freq_words = heapq.nlargest(10,word2vec,key =word2vec.get)


# In[87]:


# createing the appearence of the words in a sentence as 1 else 0 
x = []

for i in sentence:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(i) :
            vector.append(1)
        else:
             vector.append(0)
                
    x.append(vector)

# converting the list of list X in to 2d array 

x = np.asarray(x)


# In[138]:


# Calculating the TF (Total Frequence)
from collections import Counter


sentence_appearence = []
tf = {}
for i in freq_words:
    tf[i] = []
    for j in sentence:
        words_in_sent = nltk.word_tokenize(j)
        temp = Counter(words_in_sent)
        tf[i].append(temp[word]/len(words_in_sent))
        
    
for i in sentence:
    words_in_sent  = nltk.word_tokenize(i)
    temp = Counter(words_in_sent)
    word_appearence = []
    for word in freq_words:
        #word_appearence = []
        word_appearence.append(temp[word])
    
    sentence_appearence.append(word_appearence)

#sentence_appearence
tf


# In[139]:


# Calculation the IDF (Inverse Document Frequency)
import math 
idf = {}
for i in range(len(freq_words)):
    count = 0
    for j in range(len(sentence_appearence)):
        if sentence_appearence[j][i] != 0:
            count += 1
            
    idf[freq_words[i]] = np.log((len(sentence_appearence[i])/count)+1)


idf


# In[141]:


#count = 0
tf_idf = []
for i in idf:
    tf_matrix = []
    for j in tf[i]:
        score = j * idf[i]
        tf_matrix.append(score)
    tf_idf.append(tf_matrix)
    
    


# In[146]:


tf_idf_matrix = np.asarray(tf_idf)
tf_idf_matrix


# In[25]:


#Implementation of character N-gram modeling 
import random 

definition = "Global warming is a gradual increase in the overall temperature of the earth's atmosphere generally attributed to the greenhouse effect caused by increased levels of carbon dioxide,chlorofluorocarbons, and other pollutants"

# AS we have mentioned n '3' it is a try gram model 
n = 8

ngrams = {}

for i in range(0,len(definition)-n):
    if  definition[i:n+i] not in ngrams.keys():
        ngrams[definition[i:n+i]] = []
    ngrams[definition[i:n+i]].append(definition[n+i])

    
    
# Testing our ngram model, predicting the summary of the entire definition 

currentgram = definition[0:n]
result = currentgram 

for i in range(50):
    if currentgram not in ngrams.keys():
        break
    possibility = ngrams[currentgram]
    nextitem = possibility[random.randrange(len(possibility))]
    result += nextitem 
    currentgram = result[len(result)-n : len(result)]
 
print(result)


# In[43]:


n = 3

definition = "Global warming is a gradual increase in the overall temperature of the earth's atmosphere generally attributed to the greenhouse effect caused by increased levels of carbon dioxide,chlorofluorocarbons, and other pollutants"


ngrams = {}

words =nltk.word_tokenize(definition)

for i in range(len(words)-n):
    gram = ' '.join(words[i:i+n])
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(words[i+n])
    
    
currentgram = ' '.join(words[0:n])
result = currentgram
for i in range(80):
    if result not in ngrams.keys():
        break
    possibility = ngrams[currentgram]
    nextitem = possibility[random.randrange(len(possibility))]
    result += ' '+nextitem
    rwords= nltk.word_tokenize(result)
    currentgram = ' '.join(rwords[len(rwords)-n:len(rwords)])

print(result)


# In[18]:


# Latent semantic Analysis (It is used to find the category of document to which it belongs based o the content in the document)

# It is done using the SVD (Singular Value Decomposition)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

dataset = ['The amount of pollution increased day by day',
          'The concert was just great',
          'I love to see gordan ramsay cooking',
          'Google is introduction a new technology',
          'AI robots are great example of technology today',
          'All of us where singning in concert',
          'we have lunched campaign to stop pollution and global warming']

dataset = [i.lower() for i in dataset]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

lsa = TruncatedSVD(n_components = 4, n_iter = 200)
lsa.fit(X)

concept_words = {}
terms = vectorizer.get_feature_names()
for i,comp in enumerate(lsa.components_):
    components = zip(terms,comp)
    sortedterms = sorted(components, key = lambda x:x[1], reverse = True)
    sortedterms = sortedterms[:10]
    concept_words['concept'+str(i)] = sortedterms


# In[19]:


for key in concept_words.keys():
    sentence_scores =[]
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        score = 0 
        for word in words:
            for word_score in concept_words[key]:
                if word == word_score[0]:
                    score += word_score[1]
        sentence_scores.append(score)
    print("\n"+key+":")
    for sentence_score in sentence_scores:
        print(sentence_score)


# In[9]:


# Finding synonyms and antonyms of a word 

from nltk.corpus import wordnet 

synonyms = []
antonyms = []

for i in wordnet.synsets('good'):
    for j  in i.lemmas():
        if j.name() not in synonyms:
            synonyms.append(j.name())
        for k in j.antonyms():
            if k.name() not in antonyms:
                antonyms.append(k.name())
        
antonyms


# In[16]:


# Word Negation Tracking 

import nltk 

sentence = "I was not happy with the team's performance"

words = nltk.word_tokenize(sentence)

new_words = []

temp_word = ""

for word in words:
    if word == 'not':
        temp_word = 'not_'
    elif temp_word == 'not_':
        word = temp_word + word
        temp_word = ''
    if word != 'not':
        new_words.append(word)
        
sentence = ' '.join(new_words)

sentence


# In[17]:


# Negation using Antonyms

import nltk 

sentence = "I was not happy with the team's performance"

words = nltk.word_tokenize(sentence)

new_words = []

temp_word = ""


antonyms = []
for word in words:
    if word == 'not':
        temp_word = 'not_'
    elif temp_word == 'not_':
        for i in wordnet.synsets(word):
            for j  in i.lemmas():
                 for k in j.antonyms():
                    if k.name() not in antonyms:
                        antonyms.append(k.name())
        if len(antonyms) >= 1:
            word = antonyms[0]
        else:    
            word = temp_word + word
        temp_word = ''
    if word != 'not':
        new_words.append(word)
        
sentence = ' '.join(new_words)

sentence

