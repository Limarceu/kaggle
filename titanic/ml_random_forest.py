import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

#reading the dataset
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')

print(train.shape)
print(test.shape)


#Editing Key and Answer Variable
train = train.set_index(['PassengerId'])
train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

#Exploring missing values
print(train.isnull().sum())
print(test.isnull().sum())

#fill NA
train.fillna(0, inplace=True)
print(train.isnull().sum())

#Describe
print(train.describe())
#print(train.describe(include=['o']))

#Data Manipulation | Transformation
train['Women'] = np.where(train['Sex'] == 'female', 1, 0)
train['Pclass_1'] = np.where(train['Pclass'] == 1, 1, 0)
train['Pclass_2'] = np.where(train['Pclass'] == 2, 1, 0)
train['Pclass_3'] = np.where(train['Pclass'] == 3, 1, 0)

train = train.drop(['Pclass', 'Sex'], axis=1)

print(train.head())

#Training
x_train, x_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis=1), train['Survived'], test_size=0.3, random_state=0)

print(f'treino: {x_train.shape}, teste: {x_test.shape}')


rndforest = rfc(n_estimators=1000, criterion='gini', max_depth=5)

rndforest.fit(x_train, y_train)

probability = rndforest.predict_proba(train.drop('Survived', axis =1))[:,1]

classification = rndforest.predict(train.drop('Survived', axis=1))

train['probability'] = probability
train['classification'] = classification

print(train.head())