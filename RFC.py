# -*- coding: utf-8 -*-

import pandas as pd

df=pd.read_csv('Social_Network_Ads.csv')

sex=pd.get_dummies(df.Gender,drop_first=False)
df=pd.concat([df,sex],axis=1)

df=df.drop('Gender',axis=1)

X=df.iloc[:,[2,3,4,5]].values

Y=df.iloc[:,4].values

from sklearn.model_selection import train_test_split

trainx,testx,trainy,testy=train_test_split(X,Y,test_size=0.30,random_state=0)

#Accuracy====true/total
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()

trainx=SC.fit_transform(trainx)
testx=SC.fit_transform(testx)

from sklearn.ensemble import RandomForestClassifier
Classi=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
Classi.fit(trainx,trainy)


ypred=Classi.predict(testx)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(testy,ypred)

#without gender
#Accuracy: Gini=88.33%
#Accuracy: entropy=90%...... n_est=10

#with gender
#Accuracy: gini=90%
#Accuracy: entropy=89%...... n_est=10

#max. effi=92%
