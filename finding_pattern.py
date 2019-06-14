#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:23:39 2019

@author: hitesh
"""

import re 

sentence = "Hi How are you"

# this one checks for 0 or more in a string 
re.match(r".*", sentence)

# this one checks for 1 or more in a string 
re.match(r".+",sentence)

# this one checks for sentence with a-z and A-Z at the first of a sentence 
re.match(r"[a-zA-Z]+",sentence)

# this checks for H with b (0 b's or more than 0 b's)
re.match(r"Hb?",sentence)

# this checks for first a-z or A-Z in any part of the sentence eg:"1996 i was born"
re.search(r"[a-zA-Z]+", sentence)

# This checks if  the sentence starts with Hi (^ this symbol mentions first word)

if re.search(r"^How", sentence):
    print("Match")
else:
    print("Not a match")

# or 

if re.match(r"^Hi", sentence):
    print("Match")
else:
    print("Not a match")
    
# Checks for the end of the sentence, search has to be used because it checks 
# all over the sentence till the end.
    
if re.search(r"you$", sentence):
    print("Match")
else:
    print("Not a Match")
  
    
# ------------------------- Substituting the patterns in the text ----------------
    
# Replacing a word in a sentence 
sentence2 = "I love my country"

# Replacing country with Nation (Global search and replace)
print(re.sub(r"country","Nation", sentence2))

# Replacing all the letters between a-z with number 0 (note that it should be 
#mentioned as a string)
print(re.sub(r"[a-z]", "0" ,   sentence2))
#              word   replace  the sentence 

print(re.sub(r"[a-z]", "0" ,  sentence2, flags = re.I))
#               word  replace sentence   case insensitive

print(re.sub(r"[a-z]", "0" ,  sentence2, 1,         flags = re.I))
#               word  replace sentence   mentioning case insensitive
#                                        times of 
#                                        replcae
    

#------------------------- Shorthand character class ---------------------------

import re 

sentence1 = "I was born in 1947"
sentence2 = "TomoRRow is the +#$ ----- da'y of t.he class"
sentence3 = " I Love                 you"

print(re.sub(r"[-+#$\'\.]","",sentence2, flags=re.I))

print(re.sub(r"\w"," ",sentence2))#this will remove all the [a-zA-z0-9]
print(re.sub(r"\W"," ", sentence2)) #will remove all the special characters 

print(re.sub(r"\s+"," ", sentence2)) # This will remove all the large spaces 
#                                          and apply only one space to the words

print(re.sub(r"\s+[a-zA-Z]\s+"," ",sentence)) # this will remove the seperate words with no meaning 

print(re.sub(r"^\s","",sentence1)) #it remove the spaces at the begining of theword

print(re.sub(r"\s$","",sentence1)) #it remove the spaces at the end of the word

print(re.sub(r"\d","",sentence1)) #digit

print(re.sub(r" ","",sentence3,))


# =============================================================================
# Words Tokenization 
# =============================================================================

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


import nltk 
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()


for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    new_words  = [stemmer.stem(j) for j in words]
    sentences[i] = ' '.join(new_words)


sentences = nltk.sent_tokenize(paragraph)
lemmatize = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    new_words = [lemmatize.lemmatize(j) for j in words]
    sentences[i] = ' '.join(new_words)





