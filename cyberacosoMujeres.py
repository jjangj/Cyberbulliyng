# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# coding: utf-8
#Same modeling, slightly different code with similar results

# In[1]:

#Import libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer


# In[2]:

#import dataset
import pandas as pd
df1 = pd.read_csv("cleanprojectdataset.csv") #cargar datos para analizar


# In[3]:

#mostrar datos
print(df1)


# In[4]:

##Crear listas para tweets y etiquetas
Tweet = []
Labels = []

for row in df1["Tweet"]:
    #tokenize palabras
    words = word_tokenize(row)
   # print(words)
    #remove puntuaciones, eliminamos signos de puntuacion de las oraciones
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words, conexiones de palabras ejm computer for science --> computer science
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)

    for row in df1["Text Label"]:
        Labels.append(row)


# In[5]:

#Combine lists
combined = zip(Tweet, Labels)


# In[6]:

#Create bag of words
def bag_of_words(words):
    return dict([(word, True) for word in words])


# In[7]:

#Create new list for modeling
Final_Data = []
for r, v in combined:
    bag_of_words(r)
    Final_Data.append((bag_of_words(r),v))


# In[8]:


import random
random.shuffle(Final_Data)
print(len(Final_Data))


# In[9]:

