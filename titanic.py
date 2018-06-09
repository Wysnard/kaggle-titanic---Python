import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(r"dataset\train.csv")
test = pd.read_csv(r"dataset\test.csv")
combine = [train, test]
print(train.describe())
train.info()

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# train['FamilyName'] = train['Name'].str.split(',').str[0]
for dataset in combine:
	dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
	dataset['NbFamily'] = dataset['SibSp'] + dataset['Parch']

print(train[['NbFamily', 'Survived']].groupby(['NbFamily'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(pd.crosstab(train['Title'], train['Sex']))

X = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
Y = X.Survived
X = X.drop(['Survived'], axis=1)
# train['FirstName'] = train['Name'].str.split(',').str[1].str.split('.').str[1].str.strip()

from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X.Sex=labelEncoder_X.fit_transform(X.Sex)

row_index = X.Embarked.isnull()

X.loc[row_index, 'Embarked']='S'
Embarked = pd.get_dummies(X.Embarked, prefix='Embarked')
X = X.drop(['Embarked'], axis=1)
X = pd.concat([X, Embarked], axis=1)
X = X.drop(['Embarked_S'], axis=1)

X.to_csv('try.csv')
