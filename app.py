from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import torch
from PricePredictorModel import PricePredictor
import matplotlib.pyplot as plt
from time import sleep

model = PricePredictor()
model.load_state_dict(torch.load("PricePredictor.mdl")["model_state_dict"])
model.eval()

CONDITDic = {'Fair':2, 'Poor':0, 'Good':3, 'Excellent':5, 'Default':3, 'Very Good':4, 'Average':1}


def calcPrice(BATHRM=1.8030874329060018, HF_BATHRM=0.4561441308990437, HEAT=12.57708131274908, AC=0.7201982243458533, NUM_UNITS=1.1978009897905435, ROOMS=6.176532468108567, BEDRM=2.7283160436245653, AYB=1941.8852148150197, YR_RMDL=1998.2449191158516, 
EYB=1963.649229092338, STORIES=2.079705255815569, QUALIFIED=0.48203750293090664, SALE_NUM=1.6803759165024303, GBA=1704.0216186998452, STRUCT=6.496913366380177, GRADE=3.3127646473437085, CNDTN=1.9907358368238173, EXTWALL=22.418221865807894, ROOF=13.257302709619873, INTWALL=10.372404592921095, KITCHENS=1.2183845803214817, USECODE=14.238426879424084, LANDAREA=2400.347849506657, LIVING_GBA=885.2308160836482, LATITUDE=38.914819162847444, LONGITUDE=-77.01647722943854, WARD=4.569524789484975, SQUARE=2645.6188918284747, QUADRANT=2.3395339754301614):
    xData = np.zeros(29)
    xData[0] = BATHRM
    xData[1] = HF_BATHRM
    xData[2] = HEAT
    xData[3] = AC
    xData[4] = NUM_UNITS
    xData[5] = ROOMS
    xData[6] = BEDRM
    xData[7] = AYB
    xData[8] = YR_RMDL
    xData[9] = EYB
    xData[10] = STORIES
    xData[11] = QUALIFIED
    xData[12] = SALE_NUM
    xData[13] = GBA
    xData[14] = STRUCT
    xData[15] = GRADE
    xData[16] = CNDTN
    xData[17] = EXTWALL
    xData[18] = ROOF
    xData[19] = INTWALL
    xData[20] = KITCHENS
    xData[21] = USECODE   
    xData[22] = LANDAREA
    xData[23] = LIVING_GBA
    xData[24] = LATITUDE
    xData[25] = LONGITUDE
    xData[26] = WARD
    xData[27] = SQUARE
    xData[28] = QUADRANT
    xData = torch.tensor(xData).float()
    price = model(xData.view(-1, 29))
    price = int(price.item()*100000)
    st.header(f'Price: ${price}')



@st.cache(persist=True)
def read_csv_and_lowerCase_columnNames(req_cols = ["BATHRM", "ROOMS", "BEDRM", "KITCHENS", "AC", "CNDTN"]):
    df = pd.read_csv('DC_propertyOpenData.csv', encoding='latin-1', usecols=req_cols, dtype={"BATHRM": "int8", "ROOMS": "int8", "BEDRM": "int8"})
    df.columns= df.columns.str.lower()
    return df

st.title("Real Estate Price Prediction Based on Selection")
#selection box
st.write('Check out the collapsible options on the left side!')
numOfRooms = st.sidebar.slider("How many rooms?", 1, 25, value=6)
numOfBedrooms = st.sidebar.slider("How many bedrooms?", 1, 15, value=3)
numOfBathrooms = st.sidebar.slider("How many bathrooms?", 1, 10, value=2)
numOfKitchens = st.sidebar.slider("How many kitchens?", 1, 10, value=1)
squareFootage = st.sidebar.slider("How many square feet?", 250, 3500, value=2645)
AC = st.sidebar.checkbox("Have AC?", value=True)
condition = st.sidebar.selectbox('How would you like the condition of the house to be?',('Very Good','Good', 'Average'))
priceLabel = st.subheader("The estimated price of a house similar to this one is...")
if st.button('Calculate the Price'): 
    with st.spinner('Calculating price...'):
        sleep(3)
        calcPrice(BEDRM=numOfBedrooms, ROOMS=numOfRooms, BATHRM=numOfBathrooms, KITCHENS=numOfKitchens, AC=AC, CNDTN=CONDITDic[condition], SQUARE=squareFootage)
        st.balloons();
        st.success('Done!')

df = read_csv_and_lowerCase_columnNames()
df1 = df.loc[(df['bathrm']==numOfBathrooms) & (df['rooms']==numOfRooms) & (df['bedrm']==numOfBedrooms) & (df['kitchens']==numOfKitchens) & (df['ac']==AC)& (df['cndtn']==condition)]
#if df1.empty:
#    st.write("We're sorry, we don't have any houses like your preferences")
#else:
#  st.dataframe(df1)
