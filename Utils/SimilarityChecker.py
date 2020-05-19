#!/usr/bin/env python
# coding: utf-8

# In[19]:


#https://stackoverflow.com/questions/53521531/comparing-two-numpy-2d-arrays-for-similarity
#https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
#https://www.earthdatascience.org/courses/earth-analytics-bootcamp/numpy-arrays/manipulate-summarize-plot-numpy-arrays/


# In[20]:


import os
import pretty_midi
import numpy as np
import math


# In[21]:


from Utils import getDataSets


# In[22]:


workdir = "C:\\Users\\toend\\Documents\\ITU\\Thesis"


# In[23]:


X1, X2, X3, X4 = getDataSets(workdir)


# In[25]:


#PRIMARILY USED TO CHECK POSITION-WISE EQUALITY 
#(what nodes are played when, ie. overall structure but mostly amount of nodes played)
def equalityCheck(A, B):
    number_of_equal_elements = np.sum(A==B)
    total_elements = np.multiply(*A.shape)
    percentage = number_of_equal_elements/total_elements
    return percentage


# In[26]:


#USED TO CHECK THE DIFFERENCE IN THE NODES BEING PLAYED 
#(could tell something about difference in mood?)
def euclideanDistance(A, B):
    dist = np.linalg.norm(A-B)
    return dist


# In[27]:


def matchClosest(X1, X2):
    X = np.zeros((1, 3000, 128))
    Y = np.zeros((1, 3000, 128))
    X1length = X1.shape[0]
    for i in range(X1length):
        X2length = X2.shape[0]
        #maxDistance = 0
        maxScore = 0
        index = 0
        for k in range(X2length):
            #euclideanDistance = euclideanDistance(X1[i], X2[k])
            equalityScore = equalityCheck(X1[i], X2[k])
            if equalityScore == 1:
                pass
            elif equalityScore > maxScore:
                maxScore = equalityScore
                index = k
        toMatch = X1[i]       
        toMatch = np.expand_dims(toMatch, axis=0)
        X = np.append(X, toMatch, axis=0)
        match = X2[index]
        match = np.expand_dims(match, axis=0)
        Y = np.append(Y, match, axis=0)
        X2 = np.delete(X2, index, 0)
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)
    Z = list(zip(X, Y))  
    return Z


# In[28]:


def matchFurthest(X1, X2):
    X = np.zeros((1, 3000, 128))
    Y = np.zeros((1, 3000, 128))
    X1length = X1.shape[0]
    for i in range(X1length):
        X2length = X2.shape[0]
        #maxDistance = 1000
        maxScore = 1
        index = 0
        for k in range(X2length):
            #euclideanDistance = euclideanDistance(X1[i], X2[k])
            equalityScore = equalityCheck(X1[i], X2[k])
            if equalityScore < maxScore:
                maxScore = equalityScore
                index = k
        toMatch = X1[i]       
        toMatch = np.expand_dims(toMatch, axis=0)
        X = np.append(X, toMatch, axis=0)
        match = X2[index]
        match = np.expand_dims(match, axis=0)
        Y = np.append(Y, match, axis=0)
        X2 = np.delete(X2, index, 0)
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)
    Z = list(zip(X, Y))  
    return Z


# In[33]:


def getMatchedRolls(X, Y):
    #Z = matchFurthest(X1, X2)
    Z = matchClosest(X1, X2)
    return list(zip(*Z))


# In[ ]:




