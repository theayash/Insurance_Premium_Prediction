from application_logging import logger
from Processing.data_preprocessing import Preprocessing
from File_Operation import file_op
import numpy as np
import pandas as pd
import data


class Prediction_from_api:
    def __init__(self, age, sex, bmi, children, smoker, region):
        self.file_object = open("../Prediction_Logs/Prediction_api_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def prediction_api(self):
        self.log_writer.log(self.file_object, 'Start of Prediction of api....')
        InputData = pd.DataFrame(
            data=[[self.age, self.sex, self.bmi, self.children, self.smoker, self.region]],
            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        Num_Inputs = InputData.shape[0]

        DataForMl = pd.read_csv('../data/insurance.csv')
        DataForMl = DataForMl.drop(columns='expenses', axis=1)
        InputData = InputData.append(DataForMl)
        self.log_writer.log(self.file_object, f'datafor ML: {InputData.head()}')

        predictors = ['age', 'bmi', 'sex', 'children', 'smoker', 'region_northwest', 'region_southeast',
                      'region_southwest']
        # Generating the input values to the model

        preprocessor = Preprocessing(self.log_writer, self.file_object)

        InputData = preprocessor.encode_categorical_col(InputData)

        InputData = preprocessor.scaling_of_numcol(InputData)
        X = InputData[predictors].values[0:Num_Inputs]
        self.log_writer.log(self.file_object, f'Input data for model: {X}')

        file_loader = file_op.File_Operation(self.file_object)
        randomForest = file_loader.load_model('randomForest')

        prediction = randomForest.predict(X)
        # self.log_writer.log(self.file_object, f'prediction:{prediction}')
        # self.log_writer.log(self.file_object, f'prediction:{PredData.columns}')

        # TestingDataResults = pd.DataFrame(data=PredData, columns=data_input.columns)
        # PredData['Predicted_Expenses'] = np.round(prediction)
        # self.log_writer.log(self.file_object, f'prediction:{PredData.columns}')
        #
        # # path = "Prediction_Output_File/Predictions.csv"
        # PredData.to_csv(self.path, header=True,
        #                 mode='a+')  # appends result to prediction file
        # self.log_writer.log(self.file_object, 'End of Prediction')
        return prediction
