# 🏡 Housing Price Predictor

I made a simple web app using Streamlit that predicts how much a house might sell for based on some basic features you can choose.

---

## Try It Out

Here’s the live app if you want to check it out:  
[Launch Streamlit App](https://housing-price-predictor-dl6dawexdhx2vmeejeeulz.streamlit.app/)

---

## Features It Uses

- Size of the house above ground (`GrLivArea`)  
- How good the overall quality is (`OverallQual`)  
- How many cars the garage fits (`GarageCars`)  
- Total basement area (`TotalBsmtSF`)  
- The year the house was built (`YearBuilt`)

I picked these features because they seemed important and were easy to include in the app.

---

## Software I Used

- **Streamlit** for building the app  
- **Scikit-learn** with a Random Forest model to predict prices  
- **Python 3.11**  

---

## How It Works

1. You input details about the house with sliders and dropdown menus.  
2. The app uses a pre-trained machine learning model to guess the price.  
3. It shows you the estimated price right away.

---

## What I Learned

- How to work with real-world data and pick important features  
- Using `scikit-learn` to train and save a machine learning model  
- **Why Random Forest Regression was useful for this project:**  
  It works really well with structured/tabular data like housing features. It can model complex relationships between variables (like how square footage and garage size together affect price), and it’s less likely to overfit compared to a single decision tree. Plus, it doesn’t need a ton of parameter tuning to perform well, which made it a good choice for this kind of problem.  
- Building interactive apps with Streamlit  
- How to deploy a model so others can use it easily  
- The basics of cleaning and preparing data for modeling

---
