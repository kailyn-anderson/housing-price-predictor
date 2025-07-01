#!/usr/bin/env python
# coding: utf-8

import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
train.head()

# Only keep numeric columns for a quick test
train_numeric = train.select_dtypes(include=['number'])

# Drop rows with missing values
train_clean = train_numeric.dropna()

# Split features and target
# All features
# X = train_clean.drop('SalePrice', axis=1)
# y = train_clean['SalePrice']

features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
X = train_clean[features]
y = train_clean['SalePrice']

# Train model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)

# Handle missing values
train = train.dropna(axis=1, thresh=len(train) * 0.6)  # drop columns with too many missing values
train = train.fillna(method='ffill')

# Encode categoricals
train = pd.get_dummies(train)

# Remove outliers
train = train[train['GrLivArea'] < 4500]

# Align features in train/test
train, test = train.align(test, join='inner', axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"RMSE: {rmse}")

# Save the model after training (in notebook or script)
import joblib
joblib.dump(model, 'housing prices')

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('housing prices')

st.title("🏡 Housing Price Predictor")

# Sample features you want the user to input
GrLivArea = st.slider('Above Ground Living Area (sq ft)', 500, 4000, 1500)
OverallQual = st.selectbox('Overall Quality (1-10)', list(range(1, 11)))
GarageCars = st.slider('Number of Garage Cars', 0, 4, 2)
TotalBsmtSF = st.slider('Basement Area (sq ft)', 0, 3000, 800)
YearBuilt = st.slider('Year Built', 1900, 2023, 2000)

# Collect features in same order as model expects
input_features = pd.DataFrame([{
    'GrLivArea': GrLivArea,
    'OverallQual': OverallQual,
    'GarageCars': GarageCars,
    'TotalBsmtSF': TotalBsmtSF,
    'YearBuilt': YearBuilt
    # Add more if needed
}])

# Predict
if st.button('Predict Price'):
    prediction = model.predict(input_features)[0]
    st.success(f"Estimated House Price: ${prediction:,.0f}")

