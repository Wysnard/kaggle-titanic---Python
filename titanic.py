import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(r"dataset\train.csv")
test = pd.read_csv(r"dataset\test.csv")
combine = [train, test]
print(train.describe())
train.info()

print(test.describe())
test.info()

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# train['FamilyName'] = train['Name'].str.split(',').str[0]
for dataset in combine:
	labelEncoder_X = LabelEncoder()
	dataset.Sex=labelEncoder_X.fit_transform(dataset.Sex)
	dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
	dataset['NbFamily'] = dataset['SibSp'] + dataset['Parch'] + 1
	dataset['Age'] /= 100

print(train[['NbFamily', 'Survived']].groupby(['NbFamily'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(pd.crosstab(train['Title'], train['Sex']))

utitle = train['Title'].unique()
uclass = train['Pclass'].unique()

for dataset in combine:
	for i in uclass:
		for u in utitle:
			dataset.loc[(dataset.Age.isnull()) & (dataset['Title'] == u) & (dataset['Pclass'] == i), 'Age'] \
			= dataset[(dataset['Title'] == u) & (dataset['Pclass'] == i)]['Age'].median()

X_train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
Y_train = X_train.Survived
X_train = X_train.drop(['Survived'], axis=1)

X_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

X_test['Fare'].fillna(test['Fare'].dropna().median(), inplace = True)

row_index = X_train.Embarked.isnull()

X_train.loc[row_index, 'Embarked']='S'

Embarked = pd.get_dummies(X_train.Embarked, prefix='Embarked')
X_train = X_train.drop(['Embarked'], axis=1)
X_train = pd.concat([X_train, Embarked], axis=1)
X_train = X_train.drop(['Embarked_S'], axis=1)

Embarked = pd.get_dummies(X_test.Embarked, prefix='Embarked')
X_test = X_test.drop(['Embarked'], axis=1)
X_test = pd.concat([X_test, Embarked], axis=1)
X_test = X_test.drop(['Embarked_S'], axis=1)

# -------------------------------

Title = pd.get_dummies(X_train.Title, prefix='Title')
X_train = X_train.drop(['Title'], axis=1)
X_train = pd.concat([X_train, Title], axis=1)

print(X_train)

Title = pd.get_dummies(X_test.Title, prefix='Title')
X_test = X_test.drop(['Title'], axis=1)
X_test = pd.concat([X_test, Title], axis=1)

print(X_test)

X_train.to_csv('try.csv')
