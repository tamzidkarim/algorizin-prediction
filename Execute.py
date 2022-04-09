from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer, make_column_selector


model = pickle.load(open('algorizin.pkl', 'rb'))  # opening pickle file in read mode

app = Flask(__name__,template_folder='template')  # initializing Flask app


@app.route("/",methods=['GET'])
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        d1 = request.form['Reason']

        d2 = request.form['Relocation']

        d3 = request.form['Jobskill']

        d4 = request.form['techskill']
        

        dic = {'A1':[int(d3)],'A2':[int(d4)]}
        arr=pd.DataFrame(dic)
        
        
        print(arr)
        pred = model.predict(arr)
        return render_template('display.html', prediction_text= pred)
        
    else:
        return render_template('display.html', prediction_text='Please provide all necessary informations')


#app.run(host="0.0.0.0")            # deploy
app.run(debug=True)                # run on local system
