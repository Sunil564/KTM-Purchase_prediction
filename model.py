#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle


df=pd.read_excel('KTM.xlsx')
print(df.head())
print('\n')
print('\n')
print('\n')
print(df.info())

X = df.iloc[:, :-1]

Y = df.iloc[:,-1]

print(Y)



print(df['Occupation'].unique(),"\n")
print(df['Phone Type'].unique(),"\n")
print(df['Current Bike'].unique(),"\n")
print(df['Relationship'].unique(),"\n")
print(df['Gender'].unique(),"\n")
print(df['Response'].unique(),"\n")



def Occupation_to_int(word):
    word_dict = {'Professional':3, 'Self Employed':2, 'Unemployed':1, 'Student':4}
    return word_dict[word]

def Phone_type_to_int(word):
    word_dict = {'Average':2, 'Low End':1, 'High End':3}
    return word_dict[word]

def Currentbike_to_int(word):
    word_dict = {'180 to 220':4, 'No Bike':1, '220 and Above':5,'125 to 180':3, 'Below 125':2}
    return word_dict[word]

def relationship_to_int(word):   
    word_dict = {'Complicated':3, 'Single':2, 'Married':1,'Committed':4}
    return word_dict[word]

def gender_to_int(word):   
    word_dict = {'Female':1, 'Male ':2}
    return word_dict[word]

def Response_to_int(word):   
    word_dict = {'Not purchased':1, 'Purchased':2}
    return word_dict[word]


X['Occupation'] = X['Occupation'].apply(lambda x : Occupation_to_int(x))

X['Phone Type'] = X['Phone Type'].apply(lambda x :Phone_type_to_int(x))

X['Current Bike'] = X['Current Bike'].apply(lambda x : Currentbike_to_int(x))

X['Relationship'] = X['Relationship'].apply(lambda x : relationship_to_int(x))

X['Gender'] = X['Gender'].apply(lambda x : gender_to_int(x))

Y=Y.apply(lambda x : Response_to_int(x)) 


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


#logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)



pickle.dump(classifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
a=(model.predict_proba([[22,1,1,1,2,1]]))
a


for i in a:
    print(i[1])





