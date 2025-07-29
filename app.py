import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import numpy as np

# Set page config
st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="centered")

# App Title
st.title("\U0001F697 Car Price Prediction App")
st.markdown("""
#### Predict the **selling price** of a used car based on specifications.
Enter the details below:
""")

# Input Fields
car_name = st.text_input("Car Name", "Maruti Swift Dzire")
year = st.slider("Year of Manufacture", 2000, 2024, 2017)
present_price = st.number_input("Present Price (in Lakh ₹)", min_value=0.0, step=0.1, value=5.5)
driven_kms = st.number_input("Driven Kilometers", min_value=0, step=500, value=25000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Previous Owners", [0, 1, 2, 3])

# Predict Button
if st.button("Predict Selling Price"):
    # Sample input as DataFrame
    input_df = pd.DataFrame([{
        'Car_Name': car_name,
        'Year': year,
        'Present_Price': present_price,
        'Driven_kms': driven_kms,
        'Fuel_Type': fuel_type,
        'Selling_type': selling_type,
        'Transmission': transmission,
        'Owner': owner
    }])

    # Dummy training to simulate model 
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    # Sample dataset structure
    df = pd.read_csv("car data.csv")
    X = df[['Car_Name', 'Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
    y = df['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(max_features=50), 'Car_Name'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Fuel_Type', 'Selling_type', 'Transmission']),
        ('num', StandardScaler(), ['Year', 'Present_Price', 'Driven_kms', 'Owner'])
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    predicted_price = pipeline.predict(input_df)[0]

    st.success(f"Estimated Selling Price: ₹ {round(predicted_price, 2)} Lakh")
    st.markdown("---")
    st.markdown("**Note**: This is a demo app. Results may vary depending on model accuracy.")
