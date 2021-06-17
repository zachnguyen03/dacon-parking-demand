import pandas as pd
import numpy as np

#Import csv files into DataFrames
train_df = pd.read_csv('./dataset/train.csv')
age_gender_df = pd.read_csv('./dataset/age_gender_info.csv')
test_df = pd.read_csv('./dataset/test.csv')

#Rename columns to English
cols = ['Just Code', 'Total number of households', 'Rental building classification', 'Area', 'Supply type', 'Dedicated Area', 'Num household by exclusive area',
        'Free singer', 'Identity', 'Rent deposit', 'Rent', 'Subway stations within 10-min walk', 'Bus stops within 10-min walk','Parking spaces within complex', 'Registered vehicles']
train_df.columns = cols
print(train_df.head())

#Some EDA
train_df.info()
null_cols = ['Rent deposit', 'Rent', 'Subway stations within 10-min walk', 'Bus stops within 10-min walk']

age_gender_df.info()

train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

categorical_cols = ['Just Code', 'Rental building classification', 'Area', 'Supply type', 'Identity', 'Rent deposit', 'Rent']

#Check missing values
train_df.isna().sum()
test_df.isna().sum()

train_df['Dedicated Area'] = train_df['Dedicated Area'] //5 *5
test_df['Dedicated Area'] = train_df['Dedicated Area'] //5 *5

#Prepare x_train and y_train
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

X_train = train_df.iloc[:,1:-1]
y_train = train_df.iloc[:,-1]
X_test = test_df.iloc[:,1:]

#Import model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

building_class_dummies = pd.get_dummies(train_df['Rental building classification'])
building_class_dummies.columns = ['Apartment', 'Storage']
X_train = X_train.drop('Rental building classification', axis=1)
X_train = X_train.join(building_class_dummies)