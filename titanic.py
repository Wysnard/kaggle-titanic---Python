import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

train = pd.read_csv(r"dataset\train.csv")
train['Source'] = 0
train['Survived'] = train['Survived'].replace(0, -1)
result = train.Survived
train = train.drop(['Survived'], axis=1)
test = pd.read_csv(r"dataset\test.csv")
test['Source'] = 1
all_df = pd.concat([train, test])

all_df['Title'] = all_df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
all_df['NbFamily'] = all_df['SibSp'] + all_df['Parch'] + 1
all_df['Embarked'].fillna('S', inplace=True)
all_df['Cabin'].fillna('Z', inplace=True)
all_df['Cabin'] = all_df['Cabin'].str[0]
all_df['TicketSize'] = all_df['Ticket'].map(all_df.groupby('Ticket')['PassengerId'].count())
utitle = all_df['Title'].unique()
uclass = all_df['Pclass'].unique()
for i in uclass:
	all_df['Fare'].fillna(all_df.loc[all_df['Pclass'] == i]['Fare'].dropna().median(), inplace = True)
	for u in utitle:
		all_df.loc[(all_df.Age.isnull()) & (all_df['Title'] == u) & (all_df['Pclass'] == i), 'Age'] \
		= all_df[(all_df['Title'] == u) & (all_df['Pclass'] == i)]['Age'].median()
all_df = pd.get_dummies(all_df, columns=['Title', 'Pclass', 'Embarked', 'Cabin'])
all_df = all_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
all_df = all_df.apply(LabelEncoder().fit_transform)

all_df.to_csv('try.csv')

train_df = all_df[all_df['Source'] == 0].drop(['Source'], axis=1).values
train_result = result.values
test_df = all_df[all_df['Source'] == 1].drop(['Source'], axis=1).values

x = train_df
y = train_result
z = test_df

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=58, input_dim=all_df.shape[1] - 1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=29, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='relu'))

model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(100001):
    cost = model.train_on_batch(x, y)
    if step % 10000 == 0:
        print('step', step, 'train cost:', cost)

# predict
test_predict = model.predict(z)

# Generate Submission File
test_predict = np.where(test_predict>0,1,0)
NNSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': test_predict.ravel() })
NNSubmission.to_csv("NNSubmission.csv", index=False)
