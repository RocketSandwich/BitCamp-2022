from cmath import nan
from tokenize import String
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.stats import zscore

columnsToDrop = ['FIREPLACES', 'Unnamed: 0', 'SALEDATE', 'BLDG_NUM', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM', 'FULLADDRESS', 'CITY', 'STATE', 'ZIPCODE', 'NATIONALGRID', 'X', 'Y', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD', 'CENSUS_TRACT', 'CENSUS_BLOCK', 'STYLE']
columnsToFill = ['GBA', 'BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL', 'EYB', 'LIVING_GBA', 'STORIES', 'PRICE', 'SALE_NUM', 'HEAT', 'AC', 'QUALIFIED', 'STRUCT', 'GRADE', 'CNDTN', 'EXTWALL', 'ROOF', 'INTWALL', 'KITCHENS', 'USECODE', 'LANDAREA', 'LATITUDE', 'SQUARE', 'QUADRANT', 'WARD']

df = pd.read_csv("DCHousingData.csv", low_memory=False)

for col in columnsToDrop:
    df.drop([col], axis = 1, inplace = True)

heatDic = { "Wall Furnace" : 8,
            'Gravity Furnac' : 1,
            'Water Base Brd' : 10,
            'Air Exchng' : 6,
            "Forced Air" : 13,
            'Evp Cool' : 3,
            'Ind Unit' : 4,
            "Warm Cool" : 14,
            'Hot Water Rad' : 12,
            'Electric Rad' : 5,
            'Air-Oil' : 7,
            'Elec Base Brd' : 9,
            'Ht Pump' : 11,
            'No Data' : nan }

ACDic = { "Y" : 1,
          "N" : 0,
          "0": 0}

QUALDic = {
            "Q" : 1,
            "U" : 0}

STRUCTDic = {
    "Town Inside" : 2,
    "Row Inside" : 6,
    "Vacant Land" : 0,
    "Row End" : 5,
    "Default" : 4,
    "Semi-Detached" : 7,
    "Multi" : 3,
    "Town End" : 1,
    "Single": 8}

GRADEDic = {
 'Superior' : 6,
 'Fair Quality' : 1,
 'No Data' : nan,
 'Excellent' : 5,
 'Good Quality' : 3,
 'Exceptional-D' : 7,
 'Low Quality' : 0,
 'Above Average' : 4,
 'Very Good' : 5,
 'Exceptional-A' : 8,
 'Average' : 2,
 'Exceptional-B' : 10,
 'Exceptional-C' : 9,
}

CONDITDic = {'Fair':2, 'Poor':0, 'Good':3, 'Excellent':5, 'Default':3, 'Very Good':4, 'Average':1}

WALLDic = {
 'Shingle':20,
 'Common Brick':24,
 'Concrete Block':9,
 'Face Brick':5,
 'Brick Veneer':17,
 'Rustic Log':1,
 'Stone/Siding':15,
 'Stucco Block':7,
 'Brick/Stucco':8,
 'SPlaster':0,
 'Concrete':10,
 'Hardboard':11,
 'Stone':22,
 'Stucco':21,
 'Vinyl Siding':14,
 'Aluminum':6,
 'Adobe':2,
 'Wood Siding':23,
 'Brick/Siding':18,
 'Brick/Stone':19,
 'Plywood':4    ,
 'Metal Siding':12,
 'Stone Veneer':13,
 'Stone/Stucco':16,
 'Default':3}

ROOFDic = {
'Slate':14,
'Shingle':7,
'Water Proof':2,
'Metal- Pre':6,
'Typical':4,
'Concrete Tile':3,
'Wood- FS':1,
'Concrete':0,
'Metal- Cpr':8,
'Shake':10,
'Built Up':15,
'Clay Tile':11,
'Metal- Sms':12,
'Comp Shingle':13,
'Composition Ro':5,
'Neopren':9}

INTWALLDic = {
    'Vinyl Comp':0,
 'Resiliant':2,
 'Terrazo':3,
 'Vinyl Sheet':1,
 'Lt Concrete':5,
 'Hardwood':11,
 'Default':10,
 'Hardwood/Carp':9,
 'Parquet':6,
 'Wood Floor':7,
 'Ceramic Tile':4,
 'Carpet':8
}

WARDDic = {
'Ward 3':7,
 'Ward 4':4,
 'Ward 6':3,
 'Ward 8':6,
 'Ward 1':2,
 'Ward 7':1,
 'Ward 2':8,
 'Ward 5':5
}

QUADRANTDic = {
     'NW':3, 'NE':2, 'SW':0, 'SE':1
}

df["SQUARE"] = pd.to_numeric(df["SQUARE"].replace("PAR ", nan))


df = df.replace({"HEAT":heatDic})
df = df.replace({"AC":ACDic})
df = df.replace({"QUALIFIED":QUALDic})
df = df.replace({"STRUCT":STRUCTDic})
df = df.replace({"GRADE":GRADEDic})
df = df.replace({"CNDTN":CONDITDic})
df = df.replace({"EXTWALL":WALLDic})
df = df.replace({"ROOF":ROOFDic})
df = df.replace({"INTWALL":INTWALLDic})
df = df.replace({"WARD":WARDDic})
df = df.replace({"QUADRANT":QUADRANTDic})

df = df.fillna(df.mean(axis=0))

filtered_entries = (np.abs(zscore(df)) < 8).all(axis=1)
df = df[filtered_entries]


tempDrop = ['GBA', 'HF_BATHRM', 'NUM_UNITS', 'AYB', 'YR_RMDL', 'EYB', 'LIVING_GBA', 'STORIES', 'SALE_NUM', 'HEAT', 'QUALIFIED', 'STRUCT', 'GRADE', 'EXTWALL', 'ROOF', 'INTWALL', 'USECODE', 'LANDAREA', 'LATITUDE', 'LONGITUDE', 'SQUARE', 'QUADRANT', 'WARD']
for c in tempDrop:
    df.drop(c, axis = 1, inplace = True)

for c in df:
    plt.ylim(0, 1e7)
    plt.grid()
    plt.title(c)
    plt.plot(df[c].to_numpy(), df["PRICE"].to_numpy(), 'o')
    plt.show()


yDf = df["PRICE"]
df.drop("PRICE", axis = 1, inplace = True)

print(len(df))
for col in df:
    print(f"{col}={df[col].mean()}", end=" ")

yData = yDf.to_numpy(dtype=float)
xData = df.to_numpy(dtype=float)

with open("data.pkl", "wb") as File:
    pkl.dump((xData, yData), File)

