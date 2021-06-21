import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('./dataset/train.csv')
age_gender_df = pd.read_csv('./dataset/age_gender_info.csv')
test_df = pd.read_csv('./dataset/test.csv')
submission = pd.read_csv('./dataset/sample_submission.csv')

cols = ['Just Code', 'Total number of households', 'Rental building classification', 'Area', 'Supply type', 'Dedicated Area', 'Num household by exclusive area',
        'Free singer', 'Identity', 'Rent deposit', 'Rent', 'Subway stations within 10-min walk', 'Bus stops within 10-min walk','Parking spaces within complex', 'Registered vehicles']
train_df.columns = cols
test_df.columns = cols[:-1]
print(train_df.head())

lookup = {
'경상북도':'gyeongsangbukdo',
 '경상남도':'gyeongsangnamdo',
 '대전광역시':'daejeon',
 '경기도':'gyeonggido',
 '전라북도':'jeollabukdo',
 '강원도':'gangwondo',
 '광주광역시':'gwangju',
 '충청남도':'chungcheongnamdo',
 '부산광역시':'busan',
 '제주특별자치도':'jeju',
 '울산광역시':'ulsan',
 '충청북도':'chungcheongbukdo',
 '전라남도':'jeollanamdo',
 '대구광역시':'daegu',
 '서울특별시':'seoul',
 '세종특별자치시':'sejong'
    # needs completing...
}
train_df['Area']=train_df['Area'].map(lookup)
test_df['Area']=test_df['Area'].map(lookup)
age_gender_df['Area']=age_gender_df['Area'].map(lookup)

X_train['Area']=X_train['Area'].map(lookup)
X_test['Area']=X_test['Area'].map(lookup)

X_combined = X_train.append(X_test)
X_combined_encoded = pd.get_dummies(X_combined)
X_train_encoded = X_combined_encoded[0:423]
X_test_encoded = X_combined_encoded[423:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.array(X_train_encoded))
X_test_scaled = scaler.fit_transform(np.array(X_test_encoded))
Y, fitted_lambda = boxcox(y_train,lmbda=None)

model = RandomForestRegressor(n_jobs=-1, random_state=42)
avg_models.fit(X_train_scaled, Y)

pred_scaled = avg_models.predict(X_test_scaled)
pred_scaled = inv_boxcox(pred_scaled, fitted_lambda)
pred_scaled = [int(value) for value in pred_scaled]