from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os

#importing libraries for data preprocessing
import pandas as pd
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pandas.api.types import CategoricalDtype
import warnings
import json

# preprocessing module 
from prediction_1 import preprocess as preprocess_1
from prediction_2 import preprocess as preprocess_2


warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

app = Flask(__name__)
app.config["DEBUG"] = True
ALLOWED_EXTENSIONS = {'csv','xlsx','xls'}
app.config["UPLOAD_FOLDER"] = "static\\uploads"
basedir = os.path.abspath(os.path.dirname(__file__))
file_url = None
file_name = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html',file_url=file_url, filename=file_name)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/dataset', methods=['POST','GET'])
def dataset():
    # get the uploaded file
    if(request.method == 'POST'):
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if uploaded_file and allowed_file(uploaded_file.filename):
            global file_url, file_name
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
            file_url = app.config['UPLOAD_FOLDER'] + '/' + filename
            return render_template('dataset.html', file_url=file_url , filename=filename)
        return redirect(url_for('index'))
    if(request.method == 'GET'):
        return redirect(url_for('index'))
        
@app.route('/predict/<filename>', methods=['GET','POST'])
def predict(filename):
    df = pd.read_csv('static\\uploads\\'+filename)
    print(df.head())
    if(filename == "Product_Demand.csv"):
        MSE, MAE, MAPE, RMSE, ds, y, yhat = preprocess_1(df)
    else:
        MSE, MAE, MAPE, RMSE, ds, y, yhat = preprocess_2(df, 'UOPBLRBU', 'C9300', 'C9300-24P')
    # print(MSE, MAE, MAPE)
    # print(ds)
    # print(y)
    # print(yhat)
    # ds = (json.dumps(ds))
    # ds =(json.dumps(y))
    # print(json.dumps(yhat))
    # print(data)
    return render_template('output.html', file_url=file_url , filename=filename, MSE=MSE, MAE=MAE, MAPE=MAPE, RMSE=RMSE, ds=ds, y=y, yhat=yhat)
    
@app.route('/api/predict/<filename>', methods=['GET','POST'])
def predict_api(filename):
    df = pd.read_csv('static\\uploads\\'+filename)
    print(df.head())
    if(filename == "Product_Demand.csv"):
        MSE, MAE, MAPE, RMSE, ds, y, yhat = preprocess_1(df)
    else:
        MSE, MAE, MAPE, RMSE, ds, y, yhat = preprocess_2(df)    
    # print(MSE, MAE, MAPE )
    return jsonify({'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE, 'ds': jsonify(ds), 'y': jsonify(y), 'yhat': jsonify(yhat)})
        

if __name__ == '__main__':
    app.run(debug=True)
    
    