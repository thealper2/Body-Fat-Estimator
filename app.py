import pickle
import numpy as np
import pandas as pd
import streamlit as st

model = pickle.load(open("RandomForest.pkl", "rb"))

#density = 1.0709
#weight = 154.25
#chest = 93.1
#abdomen = 85.2
#hip = 94.5

st.title("Body Fat Estimator")
density = st.number_input("Density")
weight = st.number_input("Weight")
chest = st.number_input("Chest")
abdomen = st.number_input("Abdomen")
hip = st.number_input("Hip")

if st.button("Predict"):
	test = [[density, weight, chest, abdomen, hip]]
	pred = model.predict(test)[0].round(2)
	print(pred)
	st.success("Body Fat : " + str(pred))
