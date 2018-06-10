import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils

train = pd.read_csv(r"dataset\train.csv")
train['Source'] = 0
result = train['Survived']
train = train.drop(['Survived'], axis=1)
test = pd.read_csv(r"dataset\test.csv")
test['Source'] = 1
all_df = pd.concat([train, test])

all_df['Title'] = all_df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
all_df['NbFamily'] = all_df['SibSp'] + all_df['Parch'] + 1
all_df['TicketSize'] = all_df['Ticket'].map(all_df.groupby('Ticket')['PassengerId'].count())
all_df['NoAge'] = np.where(all_df['Age'].isnull(), 1, 0)
all_df['NoEmbarked'] = np.where(all_df['Embarked'].isnull(), 1, 0)
ageGroup = all_df.groupby('Title')['Age'].mean()
fareGroup = all_df.groupby('Pclass')['Fare'].median()
v = {'Age' : all_df['Title'].map(ageGroup), 'Fare' : all_df['Pclass'].map(ageGroup) \
	, 'Embarked' : 'S', 'Cabin' : 'Z'}
all_df.fillna(value=v, inplace=True)
all_df['Cabin'] = all_df['Cabin'].str[0]
all_df['Sex'] = np.where(all_df['Sex'] == 'male', 1, 0)

all_df = pd.get_dummies(all_df, columns=['Title', 'Pclass', 'Embarked', 'Cabin'])
all_df = all_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
all_df = all_df.apply(LabelEncoder().fit_transform)

all_df.to_csv('try.csv')

batch_size = 128
epochs = 10000

x_test = all_df[all_df['Source'] == 1]
x_train = all_df[all_df['Source'] == 0]
y_train = result
y_train.to_csv('y_train.csv')
dim_output = np.max(y_train) + 1
print(y_train)
print(dim_output)
y_train = np_utils.to_categorical(y_train, dim_output)
np.savetxt("category.csv", y_train, delimiter=";")

model = Sequential()

model.add(Dense(256, input_dim=all_df.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(dim_output, activation='softmax'))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit(
	x_train, y_train
	, batch_size=batch_size
	, nb_epoch=epochs
	, verbose=1
)

loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print('loss: ', loss)
print('accuracy: ', accuracy)
print()
pred_test = model.predict(x_test)
pred_test = pred_test[:, 1]
pred_test = np.where(pred_test >= 0.5, 1, 0)
print(y_train)
pred = pd.DataFrame()
pred['PassengerId'] = test['PassengerId']
pred['Survived'] = pred_test
print(pred)
pred.to_csv('titanic_pred.csv', index=False)
