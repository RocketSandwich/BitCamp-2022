import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import matplotlib.pyplot as plt

# @st.cache(persist=True)
# def read_csv_and_lowerCase_columnNames():
#     df = pd.read_csv('DC_propertyOpenData.csv',encoding='latin-1', low_memory = False)
#     df.columns= df.columns.str.lower()
#     return df
# read_csv_and_lowerCase_columnNames()
df = pd.read_csv('DC_propertyOpenData.csv',encoding='latin-1')
st.title("Real Estate Price Prediction based on selection")
#selection box
st.write("Price prediction based on your preferences")
rm = st.sidebar.slider("How many rooms?", 0, 48)
bedrm = st.sidebar.slider("How many bedrooms?", 0, 24)
bathrm = st.sidebar.slider("How many bathrooms?", 0, 14)
ktch = st.sidebar.slider("How many kitchens?", 0, 14)
AC = st.sidebar.checkbox("Have AC?")
condition = st.sidebar.selectbox('How would you like the condition of the house to be?',('Very Good','Good', 'Average'))

df1 = df.loc[(df['BATHRM'] == bathrm) & (df['ROOMS'] == rm)& (df['BEDRM']==bedrm)& (df['KITCHENS']==ktch)& (df['AC']==AC)& (df['CNDTN']==condition)]
if df1.empty:
    st.write("We're sorry, we don't have any houses like your preferences")
else:
  st.dataframe(df1)
