# 🏡 Housing Price Predictor

A simple and interactive web app built with [Streamlit](https://streamlit.io/) that predicts the sale price of a house based on user-selected property features.

![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-orange)

---

## ✨ Demo

Try the live app: [🚀 Launch Streamlit App](https://housing-price-predictor-dl6dawexdhx2vmeejeeulz.streamlit.app/)  

---

## 🔍 Features Used for Prediction

- **Above Ground Living Area** (`GrLivArea`)
- **Overall Quality** (`OverallQual`)
- **Garage Capacity** (`GarageCars`)
- **Total Basement Area** (`TotalBsmtSF`)
- **Year Built** (`YearBuilt`)

These features were selected for their strong correlation with house price and simplicity of input.

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Scikit-learn (RandomForestRegressor)
- **Language**: Python 3.11
- **Model Type**: Regression

---

## 🧠 How It Works

1. User inputs property details using sliders and dropdowns.
2. A pre-trained machine learning model predicts the price based on the inputs.
3. Prediction is instantly displayed on the app.

---
