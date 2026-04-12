import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

st.title("Simple ML Prediction App")

value = st.number_input("Enter a value", min_value=0.0, step=1.0)

if st.button("Predict"):
    prediction = model.predict(np.array([[value]]))[0]
    st.write(f"Prediction: {prediction}")
