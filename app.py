import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
df = pd.read_csv("Car Dataset Processed.csv")
df.columns = df.columns.str.strip()   # remove accidental spaces

# Normalize insurance values
df['insurance_validity'] = df['insurance_validity'].replace({
    'Third Party insurance': 'Third Party'
})

# Encoding dictionaries
d1 = {'Comprehensive': 0, 'Third Party': 1, 'Zero Dep': 2, 'Not Available': 3}
d2 = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
d3 = {'Manual': 0, 'Automatic': 1}
d4 = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth Owner': 4, 'Fifth Owner': 5}

# Apply encodings
df['insurance_validity'] = df['insurance_validity'].map(d1)
df['fuel_type'] = df['fuel_type'].map(d2)
df['transmission'] = df['transmission'].map(d3)
df['ownsership'] = df['ownsership'].map(d4)

# Handle missing values safely
df['insurance_validity'] = df['insurance_validity'].fillna(0)
df['fuel_type'] = df['fuel_type'].fillna(0)
df['transmission'] = df['transmission'].fillna(0)
df['ownsership'] = df['ownsership'].fillna(1)
df['kms_driven'] = df['kms_driven'].fillna(df['kms_driven'].median())
df['price(in lakhs)'] = df['price(in lakhs)'].fillna(df['price(in lakhs)'].median())

# Features and target
X = df[['insurance_validity', 'fuel_type', 'kms_driven', 'ownsership', 'transmission']]
y = df['price(in lakhs)']

# Train model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# Streamlit UI
st.title("🚗 Car Price Prediction App")

# User name input
user_name = st.text_input("Enter your name:")

# User inputs for car details
insurance_validity = st.selectbox("Insurance Validity", options=list(d1.values()),
                                  format_func=lambda x: {0:"Comprehensive",1:"Third Party",2:"Zero Dep",3:"Not Available"}[x])
fuel_type = st.selectbox("Fuel Type", options=list(d2.values()),
                         format_func=lambda x: {0:"Petrol",1:"Diesel",2:"CNG"}[x])
transmission = st.selectbox("Transmission", options=list(d3.values()),
                            format_func=lambda x: {0:"Manual",1:"Automatic"}[x])
ownership = st.selectbox("Ownership", options=list(d4.values()),
                         format_func=lambda x: {1:"First Owner",2:"Second Owner",3:"Third Owner",4:"Fourth Owner",5:"Fifth Owner"}[x])
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)

# Prediction
if st.button("Predict Price"):
    test = [[insurance_validity, fuel_type, kms_driven, ownership, transmission]]
    yp = model.predict(test)[0]
    if user_name.strip():
        st.success(f"{user_name}, the predicted car price is: ₹ {yp:.2f} lakhs")
    else:
        st.success(f"Predicted Car Price: ₹ {yp:.2f} lakhs")
