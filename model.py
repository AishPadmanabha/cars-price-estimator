# !/usr/bin/env python
# coding: utf-8

# # **Car Price Estimator**

# The goal of the problem statement is, given specifications of used cars, we analyse the specifications and use them to predict the price of the car. We will walk you through the following phases for predicting the price:
# 1. Data understanding
# 2. Data preprocessing
# 3. Regression
# 4. Validation

# ### **Section I: Data Understanding**
# 
# #### **Importing dependencies**

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump
import pandas as pd
import json

pd.set_option("max_columns", 150)
pd.set_option("max_rows", 150)

# #### **Load the data**
# 
# For the first part of this notebook, you'll need the 'vehicles.csv' downloaded, it is present in the data folder. The cell below will load the data from the vehicles.csv dataset into a dataframe
# 
# RAW_DATA_PATH - provides the path to the directory that contains the dataset
# RAW_DATA - dataframe that contains the loaded data present in the dataset


RAW_DATA_PATH = "data/vehicles.csv"
RAW_DATA = pd.read_csv(RAW_DATA_PATH)
print('data has been loaded, manipulations will begin now..')

# #### **About the data**
# 
# The vehicles.csv file contains data about used trucks and cars within USA collected from craigslist. The dataset can be found on kaggle at the link below:
# https://www.kaggle.com/austinreese/craigslist-carstrucks-data#vehicles.csv
# 
# The dataset contains 25 columns, here are the columns and what we interpret of them:
# 1. id: identification number of the vehicle
# 2. url: the craigslist link to the vehicle
# 3. region: the part of the country the vehicle belongs to within USA
# 4. region_url: generic link that points to the region's craiglist
# 5. price: cost listed for the vehicle
# 6. year: manufacturing year of the vehicle
# 7. manufacturer: company taht produced the vehicle
# 8. model: specific type of vehicle produced the manufacturer
# 9. condition: current state of the vehicle
# 10. cylinders: number of cylinders
# 11. fuel: type of fuel
# 12. odometer: distance travelled by the vehicle
# 13. title_status: (?)
# 14. transmission: type of transmission system of the vehicle
# 15. vin: (?)
# 16. drive: (?)
# 17. size: size of the vehicle
# 18. type: configuration of the vehicle
# 19. paint_color: color of the vehicle's body
# 20. image_url: craiglist link that provides an image for the vehicle
# 21. description: brief description of the vehicle
# 22. county: county that the vehicle belongs to
# 23. state: state that the vehicle belongs to
# 24. lat: latitude that the vehicle belongs to
# 25. long: longitude that the vehicle belongs to

# ## **Section II: Data Pre-processing**
# 
# We start by analysing the description of all the features. Next, we analyse attributes with missing values and the number of missing values by performing a DF.describe().
# Here are our observations:
# 1. it is clear that we don't need the county attribute as it has no values to offer. We can get rid of the column
# 2. factors like url, image_url, region_url, vin, description, lat and lon aren't primary factors and can be removed
manufacturer_dict = {
    'volkswagen': 0,
    'honda': 1,
    'ford': 2,
    'nissan': 3,
    'jeep': 4,
    'gmc': 5,
    'hyundai': 6,
    'chevrolet': 7,
    'toyota': 8,
    'lexus': 9,
    'ram': 10,
    'dodge': 11,
    'mini': 12,
    'audi': 13,
    'lincoln': 14,
    'mazda': 15,
    'mercedes-benz': 16,
    'acura': 17,
    'subaru': 18,
    'bmw': 19,
    'cadillac': 20,
    'volvo': 21,
    'buick': 22,
    'saturn': 23,
    'kia': 24,
    'rover': 25,
    'infiniti': 26,
    'mitsubishi': 27,
    'chrysler': 28,
    'jaguar': 29,
    'mercury': 30,
    'pontiac': 31,
    'harley-davidson': 32,
    'tesla': 33,
    'fiat': 34,
    'land rover': 35,
    'aston-martin': 36,
    'ferrari': 37,
    'alfa-romeo': 38,
    'porche': 39,
    'hennessey': 40
}

condition_dict = {
    'excellent': 0,
    'good': 1,
    'like new': 2,
    'fair': 3,
    'new': 4,
    'salvage': 5
}


cylinders_dict = {
    '4 cylinders': 4,
    '10 cylinders': 10,
    '6 cylinders': 6,
    '8 cylinders': 8,
    '5 cylinders': 5,
    '3 cylinders': 3,
    '12 cylinders': 12,
    'other': 0
}

fuel_dict = {
    'gas': 0,
    'other': 1,
    'diesel': 2,
    'hybrid': 3,
    'electric': 4,
}

transmission_dict = {
    'manual' : 0,
    'automatic' : 1,
    'electric': 2,
    'other' : 0
}

drive_dict = {
    '4wd' : 4,
    'fwd' : 1,
    'rwd' : 0
}

size_dict = {
    'compact' : 0,
    'mid-size' : 1,
    'full-size' : 2,
    'sub-compact' : 3
}

type_dict = {
    'hatchback' : 1,
    'sedan' : 2,
    'truck' : 3,
    'coupe' : 4,
    'SUV' : 5,
    'pickup' : 6,
    'wagon' : 7,
    'convertible' : 8,
    'van' : 9,
    'other' : 10,
    'mini-van' : 11,
    'bus' : 12,
    'offroad': 13
}

paint_color_dict = {
    'black' : 0,
    'grey' : 1,
    'white' : 2,
    'blue' : 3,
    'custom' : 4,
    'yellow' : 5,
    'silver' : 6,
    'red' : 7,
    'brown' : 8,
    'green' : 9,
    'purple': 10,
    'orange': 11
}

with open('models_dict.txt') as json_file:
    models_dict = json.load(json_file)
    
DF_BASIC = RAW_DATA[['price', 'manufacturer', 'condition', 'cylinders', 'model',
                     'fuel', 'odometer', 'transmission', 'drive', 'size', 'type', 'paint_color', 'year']]
DF_BASIC = DF_BASIC.rename(columns={'type': 'vtype'})

DF_BASIC = DF_BASIC.dropna()
DF_BASIC = DF_BASIC.drop_duplicates()
DF_BASIC = DF_BASIC[(DF_BASIC.year > 1995) ]
DF_BASIC=DF_BASIC.round({'year':0,'odometer':0})
DF_BASIC=DF_BASIC[(DF_BASIC.price > 100) & (DF_BASIC.price < 125000) ]
DF_BASIC = DF_BASIC[(DF_BASIC.odometer > 50) ]
DF_BASIC.drop(DF_BASIC.query('price<1000').query('year>2015').index , inplace=True)

DF_BASIC=DF_BASIC.reset_index(drop=True)

DF_BASIC['manufacturer'] = [manufacturer_dict[x] for x in DF_BASIC['manufacturer']]
DF_BASIC['condition'] = [condition_dict[x] for x in DF_BASIC['condition']]
DF_BASIC['cylinders'] = [cylinders_dict[x] for x in DF_BASIC['cylinders']]
DF_BASIC['fuel'] = [fuel_dict[x] for x in DF_BASIC['fuel']]
DF_BASIC['transmission'] = [transmission_dict[x] for x in DF_BASIC['transmission']]
DF_BASIC['size'] = [size_dict[x] for x in DF_BASIC['size']]
DF_BASIC['vtype'] = [type_dict[x] for x in DF_BASIC['vtype']]
DF_BASIC['paint_color'] = [paint_color_dict[x] for x in DF_BASIC['paint_color']]
DF_BASIC['drive'] = [drive_dict[x] for x in DF_BASIC['drive']]
DF_BASIC['model'] = [models_dict[x] for x in DF_BASIC['model']]

DF_BASIC = pd.get_dummies(DF_BASIC)

print('data manipulations are done, wait for training to begin')

# ## **Section 3: Perform Regression**
# 
# We start by preparing the cleaned data to make it fit for training. We first define our X (input variables) and y (outcomes). We then use the train_test_split package from scikit learn to split the data into train and test samples (80:20) resulting in X_train, X_test, Y_train, Y_test.


X = DF_BASIC.drop(columns='price')[:3000]
y = DF_BASIC['price'][:3000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


gb = GradientBoostingRegressor()
rf = RandomForestRegressor(n_estimators=1000, max_depth=25, random_state=25)
lr = LinearRegression()

print('training will begin now..')

gb.fit(X_train, y_train)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ## **Section 4: Saving the model**
# 
# We save our model as a joblib file. This joblib file may be imported anytime to be implemented in a service.

print('training is done, storing model')

dump(lr, 'model/lr.joblib')
dump(gb, 'model/gb.joblib')
dump(rf, 'model/rf.joblib')