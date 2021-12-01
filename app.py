import bcrypt as bcrypt
from flask import Flask
from flask_cors import CORS, cross_origin
# import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, session,send_file
import pandas as pd

from Prediction_Model.prediction_for_csv import Prediction_csv
from application_logging import logger
from Processing.data_preprocessing import Preprocessing
from File_Operation import file_op
import numpy as np
import pandas as pd
import json
import os
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from Prediction_Model import prediction_for_csv
from datetime import datetime


import pymongo

from Prediction_Model.prediction_for_input import Prediction_from_api

app = Flask(__name__)
bcrypt = Bcrypt(app)
CORS(app)

with open('config.json', 'r') as r:
    param = json.load(r)['params']

app.config['UPLOAD_FOLDER'] = param['upload_location']
app.config["MONGO_URI"] = param['server_location']

def mongoConect(dbName, collectionName):
    try:
        client = pymongo.MongoClient(app.config["MONGO_URI"])
        dataBase = client[dbName]
        collection = dataBase[collectionName]
        return dataBase, collection
    except Exception as e:
        print('Error in mongo connection',e)



@app.route("/", methods=['POST', 'GET'])
@cross_origin()
def login():
    return render_template('login.html')


@app.route("/register", methods=['POST', 'GET'])
@cross_origin()
def register():
    if request.method == 'POST':
        db, collection = mongoConect('InsuranceData', 'User_Data')
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        existing_user = collection.find_one({'email': email})
        if existing_user is None:
            hasspassword = bcrypt.generate_password_hash(password).decode('utf-8')
            collection.insert({'name': name, 'email': email, 'password': hasspassword})
            #session['name'] = name
            flash('Registered Successfully ! Please Login !','success')
            return redirect(url_for('register'))
        flash('Registration Failed ! Email Already Exists !','error')
        return render_template('register.html')

    return render_template('register.html')


@app.route("/login_validation", methods=['post', 'get'])
@cross_origin()
def login_validation():
    db, collection = mongoConect('InsuranceData', 'User_Data')

    login_user = collection.find_one({'email': request.form['email']})

    if login_user:


        if bcrypt.check_password_hash(login_user['password'], request.form['password']):
            session['email'] = request.form['email']
            return redirect(url_for('home'))
    flash('Invalid ID or password !!','error')
    return redirect(url_for('login'))


@app.route("/logout", methods=['post', 'get'])
@cross_origin()
def logout():
    if 'email' in session:
        session.pop('email',None)
        flash('You have been Logged out !!','success')
        return redirect(url_for('login'))
    return redirect(url_for('login'))

@app.route("/home", methods=['POST', 'GET'])
@cross_origin()
def home():
    if 'email' in session:
        msg = 'you are logged in as ' + session['email']
        return render_template('insurance.html',msg=msg)
    return render_template('login.html')







@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def Predict_api():
    if 'email' in session:
        msg = 'you are logged in as ' + session['email']

        if request.method == 'POST':
            data1 = float(request.form['Age'])
            data2 = request.form['Sex']
            data3 = float(request.form['BMI'])
            data5 = request.form['Smoker']
            data4 = int(request.form['Children'])
            data6 = request.form['Region']
            data7 = request.form['fn']

            InputData = pd.DataFrame(
                data=[[data1, data2, data3, data4, data5, data6]],
                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            # print(InputData.head())
            Num_Inputs = InputData.shape[0]
            # print('num of inp',Num_Inputs)

            DataForMl = pd.read_csv('data/insurance.csv')
            # print(DataForMl.head())
            DataForMl = DataForMl.drop(columns='expenses', axis=1)
            InputData = InputData.append(DataForMl)

            # self.log_writer.log(self.file_object, f'datafor ML: {InputData.head()}')

            predictors = ['age', 'bmi', 'sex', 'children', 'smoker', 'region_northwest', 'region_southeast',
                          'region_southwest']
            # Generating the input values to the model

            file_object = open("Prediction_Logs/Prediction_api_Log.txt", 'a+')
            # print(InputData.head())
            cat_df = InputData[['sex', 'smoker', 'region']].copy()
            # print(cat_df.head())

            cat_df['sex'] = cat_df['sex'].map({'female': 0, 'male': 1})
            cat_df['smoker'] = cat_df['smoker'].map(
                {'yes': 1, 'no': 0})
            # self.logger_object.log(self.file_object, f'data col value:{self.cat_df.head(5)}')
            # print(cat_df.head())
            # Using the dummy encoding to encode the categorical columns to numerical ones

            cat_df = pd.get_dummies(cat_df, drop_first=True)
            # print(cat_df)
            # self.logger_object.log(self.file_object, f'data col value:{self.cat_df.head(5)}')
            # print(InputData.head())
            InputData.drop(columns=['sex', 'smoker', 'region'], inplace=True)
            # print(InputData.head())
            # self.logger_object.log(self.file_object, f'data col value:{self.cat_df.head(5)}')
            data = pd.concat([cat_df, InputData], axis=1)

            # print(data.head())
            log_writer = logger.App_Logger()
            file_object = open("Prediction_Logs/Prediction_api_Log.txt", 'a+')
            pre = Preprocessing(log_writer, file_object)
            data = pre.scaling_of_numcol(data)
            # print(data)

            X = data[predictors].values[0:Num_Inputs]
            # self.log_writer.log(self.file_object, f'Input data for model: {X}')
            # print(X)

            file_loader = file_op.File_Operation(file_object)
            randomForest = file_loader.load_model('randomForest')

            prediction = int(randomForest.predict(X))
            prediction = np.round(prediction)

            return render_template('insurance.html', pred=f'Hi {data7} as per your information Your predicted expense is {prediction}', msg=msg)


    return redirect(url_for('login'))


@app.route("/uploadPredict", methods=['POST'])
@cross_origin()
def predict_file():
    try:
        if 'email' in session:
            msg = 'you are logged in as ' + session['email']
            if request.method == 'POST':
                f = request.files['filename']
                # f.filename = session['email']+'_'+f.filename
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
                db, collection = mongoConect('InsuranceData', 'User_File')

                now = datetime.now()
                date = now.strftime("%d/%m/%Y")
                time=now.strftime("%H:%M:%S")


                collection.insert_one({'email': session['email'],'file_name':f.filename,'date':date,'time':time})
                # print(i for i in db.User_File.find( { 'email': session['email'] }, { 'file_name':1, '_id': 0 } ))
                # print(db.collection.find())
                # print(f.filename)

                path = 'Prediction_Output_File/Predictions.csv'
                # if os.path.exists(path):
                #     os.remove(path)
                # else:
                #     print("Can not delete the file as it doesn't exists")
                # outputFile=session['email']+ '_'+f.filename
                user_name=session['email']
                predCSV = prediction_for_csv.Prediction_csv(f.filename,user_name)
                predCSV.predictionFromModel()

                PredOutputFile = predCSV.path

                return send_file(PredOutputFile, as_attachment=True)

        return render_template('login.html')
    except Exception as e:
        raise e
    # finally:
    #     dir = 'predictionFromDB/'
    #     for f in os.listdir(dir):
    #         os.remove(os.path.join(dir, f))



@app.route("/about", methods=['POST','GET'])
@cross_origin()
def about():
    if 'email' in session:
        msg = 'you are logged in as ' + session['email']

        return render_template('about.html')
    return render_template('login.html')






'''
@app.route("/predictFile", methods=['POST'])
@cross_origin()
def predict_file():
    if request.method == 'POST':
        predCSV = prediction_for_csv.Prediction_csv('Book1.csv')
        predCSV.predictionFromModel()

        return render_template('insurance.html', update='Predicted Successfully')'''


if __name__ == "__main__":
    app.secret_key = 'mysecret'
    app.run(debug=True)

