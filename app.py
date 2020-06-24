# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:31:40 2020

@author: Madhur
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('pickle_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_arr = model.predict_proba(final_features)
    prediction=np.array_split(prediction_arr,2)
#print(prediction[0][0][0])
#print(prediction[0][0][1])

    output_0 = prediction[0][0][0]*100
    output_1 = prediction[0][0][1]*100

    return render_template('index.html', prediction_text='Percentage of Customer being Zero i.e Default Loan is {}% \n Percentage of Customer being one i.e OK or GOOD is {}%'.format(output_0,output_1))


if __name__ == "__main__":
    app.run(debug=True)