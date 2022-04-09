import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import matplotlib.pyplot as plt


st.title("Real Estate Price Prediction based on selection")
df = pd.read_csv('DC_propertyOpenData.csv',encoding='latin-1')
df.columns= df.columns.str.lower()
test = df.astype(str)
# st.dataframe(test)

#Number of bathrooms
bathrmL = df['bathrm'].tolist()
maxBath =  max(list(set(bathrmL)))
minBath =  min(list(set(bathrmL)))

#Number of rooms
roomL = df['rooms'].tolist()
maxRm =  max(list(set(roomL)))
minRm =  min(list(set(roomL)))

#Number of bedrooms
BedrmL = df['bedrm'].tolist()
maxBed =  max(list(set(BedrmL)))
minBed =  min(list(set(BedrmL)))

#Number of kitchens
kitL = df['kitchens'].tolist()
maxKit =  max(list(set(kitL)))
minKit =  min(list(set(kitL)))


# selection box
st.write("Your preference")
numOfRooms = st.slider("How many rooms?", minRm, maxRm)
numOfBedrooms = st.slider("How many bedrooms?", minBed, maxBed)
numOfBathrooms = st.slider("How many bathrooms?", minBath, maxBath)
numOfKitchen = st.slider("How many kitchens?", minKit, maxKit)
AC = st.checkbox("Have AC?")
condition = st.selectbox('How would you like the condition of the house to be?',('Very Good','Good', 'Average'))

#selection df 1
selectDF1 = df.loc[(df['bathrm'] == numOfBathrooms)]
