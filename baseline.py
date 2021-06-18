import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

#Import csv files into DataFrames
train_df = pd.read_csv('./dataset/train.csv')
age_gender_df = pd.read_csv('./dataset/age_gender_info.csv')
test_df = pd.read_csv('./dataset/test.csv')
submission = pd.read_csv('./dataset/sample_submission.csv')

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

#Group each just code together
columns = ['Just Code', 'Total number of households', 'Free singer', 'Area', 'Parking spaces within complex', 'Subway stations within 10-min walk', 'Bus stops within 10-min walk']
target = 'Registered vehicles'
area_columns = []
for area in train_df['Dedicated Area'].unique():
    area_columns.append(f'Area_{area}')

new_train = pd.DataFrame()
new_test = pd.DataFrame()

for i, code in tqdm(enumerate(train_df['Just Code'].unique())):
        temp = train_df[train_df['Just Code'] == code]
        temp.index = range(temp.shape[0])
        for col in columns:
                new_train.loc[i, col] = temp.loc[0, col]

        for col in area_columns:
                area = float(col.split('_')[-1])
                new_train.loc[i, col] = temp[temp['Dedicated Area'] == area]['Num household by exclusive area'].sum()

        new_train.loc[i, 'Registered vehicles'] = temp.loc[0, 'Registered vehicles']

for i, code in tqdm(enumerate(test_df['Just Code'].unique())):
        temp = test_df[test_df['Just Code'] == code]
        temp.index = range(temp.shape[0])
        for col in columns:
                new_test.loc[i, col] = temp.loc[0, col]

        for col in area_columns:
                area = float(col.split('_')[-1])
                new_test.loc[i, col] = temp[temp['Dedicated Area'] == area]['Num household by exclusive area'].sum()

#Prepare x_train and y_train
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)
new_train = new_train.fillna(-1)
new_test = new_test.fillna(-1)

X_train = new_train.iloc[:,1:-1]
y_train = new_train.iloc[:,-1]
X_test = new_test.iloc[:,1:]

X_train = X_train.drop('Area', axis=1)
X_test = X_test.drop('Area', axis=1)

#Import model
model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

building_class_dummies = pd.get_dummies(train_df['Rental building classification'])
building_class_dummies.columns = ['Apartment', 'Storage']
X_train = X_train.drop('Rental building classification', axis=1)
X_train = X_train.join(building_class_dummies)

building_class_dummies_test = pd.get_dummies(test_df['Rental building classification'])
X_test = X_test.drop('Rental building classification', axis=1)
X_test = X_test.join(building_class_dummies_test)

pred = model.predict(X_test)
submission['num'] = pred
submission.to_csv('baseline.csv', index=False)