from File_Operation.file_op import File_Operation
from application_logging import logger
from data_loader import data_loader_prediction
from Processing.data_preprocessing import Preprocessing
from File_Operation import file_op
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

class Prediction_csv:
    def __init__(self,filename,user_name):
        self.filename = filename
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        if not os.path.exists('Prediction_Output_File/'+user_name):
            os.makedirs('Prediction_Output_File/'+user_name)

        now = datetime.now()
        date = now.strftime("%d.%m.%Y")
        time = now.strftime("%H.%M.%S")

        self.path = "Prediction_Output_File/"+user_name+'/'+'Date_'+date+'_'+'Time_'+time+'_'+filename

    def predictionFromModel(self):

        try:

            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            self.data_input = data_getter.get_data(self.filename)
            self.log_writer.log(self.file_object, f'data input col:{self.data_input.columns}')
            PredData = self.data_input.copy()
            if PredData['region'].nunique()==4:
                preprocessor = Preprocessing(self.log_writer, self.file_object)
                self.data = preprocessor.checking_missing_values(self.data_input)

                self.data = preprocessor.encode_categorical_col(self.data)

                self.data = preprocessor.scaling_of_numcol(self.data)

                file_loader = file_op.File_Operation(self.file_object)
                randomForest = file_loader.load_model('randomForest')

                prediction = randomForest.predict(self.data)
                # self.log_writer.log(self.file_object, f'prediction:{prediction}')
                self.log_writer.log(self.file_object, f'prediction:{PredData.columns}')

                # TestingDataResults = pd.DataFrame(data=PredData, columns=data_input.columns)
                PredData['Predicted_Expenses'] = np.round(prediction)
                self.log_writer.log(self.file_object, f'prediction:{PredData.columns}')

                # path = "Prediction_Output_File/Predictions.csv"
                PredData.to_csv(self.path, header=True,
                                mode='a+')  # appends result to prediction file
                self.log_writer.log(self.file_object, 'End of Prediction')
            else:
                print('lesser region')
                preprocessor = Preprocessing(self.log_writer, self.file_object)

                Num_Inputs = self.data_input.shape[0]
                DataForMl = pd.read_csv('data/insurance.csv')
                DataForMl = DataForMl.drop(columns='expenses', axis=1)
                InputData =self.data_input.append(DataForMl)
                predictors = ['age', 'bmi', 'sex', 'children', 'smoker', 'region_northwest', 'region_southeast',
                              'region_southwest']
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

                prediction = randomForest.predict(X)
                prediction = np.round(prediction)
                PredData['Predicted_Expenses'] = np.round(prediction)
                PredData.to_csv(self.path, header=True,
                                mode='a+')  # appends result to prediction file
                self.log_writer.log(self.file_object, 'End of Prediction')






        except Exception as ex:

            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        # return path

