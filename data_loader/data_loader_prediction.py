import pandas as pd
import os
import json


class Data_Getter_Pred:
    """
    This class shall  be used for obtaining the data from the source for prediction.

    Written By: Sambit Kumar Behera
    Version: 1.0
    Revisions: None

    """

    def __init__(self, file_object, logger_object):

        with open("config.json", "r") as r:
            param = json.load(r)['params']
        self.prediction_file = param['upload_location']
        #self.prediction_file = r"C:\Users\dell\PycharmProjects\InsurancePremium\predictionFromDB\Book1.csv"

        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self,filename):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By:Sambit Kumar Behera
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the Data_Getter class')
        try:
            self.prediction_file = self.prediction_file + filename
            print(self.prediction_file)
            self.data = pd.read_csv(self.prediction_file)  # reading the data file
            self.data.columns = map(str.lower, self.data.columns)
            self.logger_object.log(self.file_object,
                                   'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_data method of the Data_Getter class. Exception message: ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()
# from application_logging import logger
# data=Data_Getter_Pred(open("../Prediction_Logs/Prediction_Log.txt", 'a+'),logger.App_Logger())
# data=data.get_data('Book2_100.csv')
# print(data)
# print(os.getcwd())