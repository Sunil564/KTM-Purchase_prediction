#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# In[80]:


df=pd.read_excel('KTM.xlsx')
print(df.head())
print('\n')
print('\n')
print('\n')
print(df.info())


# In[86]:


#Moving target variable that is the price to X and all the dependent variables to y

X = df.iloc[:, :-1]

Y = df.iloc[:,-1]

print(Y)


# In[87]:


print(df['Occupation'].unique(),"\n")
print(df['Phone Type'].unique(),"\n")
print(df['Current Bike'].unique(),"\n")
print(df['Relationship'].unique(),"\n")
print(df['Gender'].unique(),"\n")
print(df['Response'].unique(),"\n")


# In[88]:


def Occupation_to_int(word):
    word_dict = {'Professional':1, 'Self Employed':2, 'Unemployed':3, 'Student':4}
    return word_dict[word]

def Phone_type_to_int(word):
    word_dict = {'Average':1, 'Low End':2, 'High End':3}
    return word_dict[word]

def Currentbike_to_int(word):
    word_dict = {'180 to 220':1, 'No Bike':2, '220 and Above':3,'125 to 180':4, 'Below 125':5}
    return word_dict[word]

def relationship_to_int(word):   
    word_dict = {'Complicated':1, 'Single':2, 'Married':3,'Committed':4}
    return word_dict[word]

def gender_to_int(word):   
    word_dict = {'Female':1, 'Male ':2}
    return word_dict[word]

def Response_to_int(word):   
    word_dict = {'Not purchased':1, 'Purchased':2}
    return word_dict[word]


# In[89]:


X['Occupation'] = X['Occupation'].apply(lambda x : Occupation_to_int(x))

X['Phone Type'] = X['Phone Type'].apply(lambda x :Phone_type_to_int(x))

X['Current Bike'] = X['Current Bike'].apply(lambda x : Currentbike_to_int(x))

X['Relationship'] = X['Relationship'].apply(lambda x : relationship_to_int(x))

X['Gender'] = X['Gender'].apply(lambda x : gender_to_int(x))

Y=Y.apply(lambda x : Response_to_int(x)) 


# In[90]:


X


# In[91]:



#splitting the data into training set and test set

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[92]:


#logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(Y_pred)


# In[93]:


#summary of the prediction made by the classifier

print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))

#Accuracy score

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_pred,Y_test)*100)    # 100 for making pecentage

#or classifier.score(X_test, Y_test)


# In[122]:



pickle.dump(classifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
a=(model.predict_proba([[22,1,1,1,1,1]]))
a


# In[123]:



for i in a:
    print(i[1])


# In[120]:




