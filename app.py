
import os
from flask import Flask,redirect,url_for,render_template,request
import pickle
import numpy as np
import torch
import pandas as pd

model = pickle.load(open('fert_LR.pkl', 'rb'))    
# model = pickle.load(open('DecisionTree.pkl', 'rb'))    

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('fert.html')

@app.route('/predict', methods = ["POST"])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template("fert.html", prediction_text = "The fertilizer required is {}".format(prediction[0]))
    # return render_template("fert.html", prediction_text = "The crop most suitable is {}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)