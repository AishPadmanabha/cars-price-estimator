import connexion
import joblib
import pandas as pd
import numpy as np
import json

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
clf = joblib.load('./model/gb.joblib')

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

def createInputData(manufacturer, model, condition, cylinders, fuel, transmission, drive, size, vtype, paint_color):
    manufacturer = manufacturer_dict.get(manufacturer)
    model = models_dict.get(model)
    condition = condition_dict.get(condition)
    cylinders = cylinders_dict.get(cylinders)
    fuel = fuel_dict.get(fuel)
    transmission = transmission_dict.get(transmission)
    drive = drive_dict.get(drive)
    size = size_dict.get(size)
    vtype = type_dict.get(vtype)
    paint_color = paint_color_dict.get(paint_color)
    return manufacturer, model, condition, cylinders, fuel, transmission, drive, size, vtype, paint_color

# Implement our predict function
def predict(manufacturer, model, condition, cylinders, fuel, odometer, transmission, drive, size, vtype, paint_color, year):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    manufacturer, model, condition, cylinders, fuel, transmission, drive, size, vtype, paint_color = createInputData(manufacturer, model, condition, cylinders, fuel, transmission, drive, size, vtype, paint_color)
    prediction = clf.predict([[manufacturer, model, condition, cylinders, fuel, odometer, transmission, drive, size, vtype, paint_color, year]])

    print(prediction[0])
    # Return the prediction as a json
    return {"prediction": str(prediction[0])}

# Read the API definition for our service from the yaml file
app.add_api("api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
