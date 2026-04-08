

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("house_data.csv")

# CLEAN
for col in ['city','areaname','propertytype']:
    data[col] = data[col].str.lower().str.strip()

# ENCODERS
city_enc = LabelEncoder()
area_enc = LabelEncoder()
type_enc = LabelEncoder()

data['city'] = city_enc.fit_transform(data['city'])
data['areaname'] = area_enc.fit_transform(data['areaname'])
data['propertytype'] = type_enc.fit_transform(data['propertytype'])

# FEATURES
X = data[['city','areaname','areasqft','bedrooms','propertytype']]
y = data['price']

# MODEL
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# SAVE
pickle.dump((model, city_enc, area_enc, type_enc), open("model.pkl","wb"))

print("Model trained successfully")