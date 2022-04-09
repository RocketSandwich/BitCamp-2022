import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import matplotlib.pyplot as plt

@st.cache(persist=True)
def read_csv_and_lowerCase_columnNames(df):
    df = pd.read_csv('DC_propertyOpenData.csv',encoding='latin-1', low_memory = False)
    df.columns= df.columns.str.lower()
    return df

st.title("Real Estate Price Prediction based on selection")
#selection box
st.write("Your preference")
numOfRooms = st.slider("How many rooms?", 0, 48)
numOfBedrooms = st.slider("How many bedrooms?", 0, 24)
numOfBathrooms = st.slider("How many bathrooms?", 0, 14)
numOfKitchen = st.slider("How many kitchens?", 0, 14)
AC = st.checkbox("Have AC?")
condition = st.selectbox('How would you like the condition of the house to be?',('Very Good','Good', 'Average'))

#selection df 1
# selectDF1 = df.loc[(df['bathrm'] == numOfBathrooms) &(df['rooms'] == numOfRooms) ]
# st.dataframe(selectDF1)
# df1 = pd.DataFrame(df, columns= ['latitude','longitude'])
# st.map(df1)
