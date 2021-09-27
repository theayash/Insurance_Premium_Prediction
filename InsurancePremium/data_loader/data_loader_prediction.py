import pandas as pd
import os


class Data_Getter_Pred:
    """
    This class shall  be used for obtaining the data from the source for prediction.

    Written By: Sambit Kumar Behera
    Version: 1.0
    Revisions: None

    """

    def __init__(self, file_object, logger_object):
        self.prediction_file = 'predictionFromDB/'
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
            self.data = pd.read_csv(self.prediction_file)  # reading the data file
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
