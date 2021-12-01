from application_logging import logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessing:

    def __init__(self,logger_object,file_object):
        self.logger_object = logger_object
        self.file_object = file_object

    def checking_missing_values(self,data):
        """
                                       Method Name: is_null_present
                                       Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                       Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                               returns the list of columns for which null values are present.
                                       On Failure: Raise Exception

                                       Written By: Sambit Kumar Behera
                                       Version: 1.0
                                       Revisions: None

                               """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.cols = data.columns
        self.null_counts = data.isna().sum()
        self.null_cols = []
        self.con_cols = ['age', 'bmi']
        self.cat_cols = ['sex','children','smoker','region']
        try:
            if data.isnull().values.any():
                for i in range(len(self.null_counts)):
                    if self.null_counts[i] > 0:
                        self.null_cols.append(self.cols[i])

                for i in self.null_cols:
                    if i in self.con_cols:

                        data[i]=data[i].replace(np.NAN, data[i].mean())


                    else:

                        data[i] = data[i].fillna(data[i].value_counts().index[0])


            self.logger_object.log(self.file_object,
                                   'Data has been treated')
            return  data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: Sambit Kumar Behera
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def outier_treatment(self,data,columns=None):

        self.logger_object.log(self.file_object, 'Entered the outlier treatment method of the Preprocessor class')

        self.data=data
        self.col=columns

        try:
            for col in self.col:

                self.uppper_boundary = data[col].mean() + 3 * data[col].std()
                self.lower_boundary = data[col].mean() - 3 * data[col].std()
                #print(lower_boundary), print(uppper_boundary), print(data['age'].mean())
                self.logger_object.log(self.file_object,
                                       f'Upper limit :{self.uppper_boundary}, lower limit:{self.lower_boundary} set for column {col}')
                try:
                    data.loc[data[col] > self.uppper_boundary, col] = self.uppper_boundary
                except Exception as e:
                    self.logger_object.log(self.file_object, 'getting error %s'%e)

                self.logger_object.log(self.file_object, f'Column {col} has been treated.')

            return data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in Outlier TReatment Method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Outlier treatment failed. Exited the is_null_present method of the Preprocessor class')
            return 'error: %s'%e

    def scaling_of_numcol(self,data):
        """
                                                                Method Name: scaling_of_Numcol
                                                                Description: This method scales the numerical values using the Standard scaler.
                                                                Output: A dataframe with scaled values
                                                                On Failure: Raise Exception

                                                                Written By: sambit kumar behera
                                                                Version: 1.0
                                                                Revisions: None
                                             """
        self.logger_object.log(self.file_object,
                               'Entered the scaling_of_Numcol method of the Preprocessor class')

        self.data = data
        self.num_df = self.data[['age','bmi']]

        try:

            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns, index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()

    def encode_categorical_col(self,data):
        """
                                                Method Name: encode_categorical_col
                                                Description: This method encodes the categorical values to numeric values.
                                                Output: dataframe with categorical values converted to numerical values
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.logger_object.log(self.file_object, 'Entered the encode_categorical_columns method of the Preprocessor class')

        self.data=data
        try:
            self.cat_df = self.data[['sex','smoker','region']].copy()
            self.logger_object.log(self.file_object, f'data:{self.cat_df.columns}')
            self.cat_df['sex'] = self.cat_df['sex'].map({'female': 0, 'male': 1})
            self.cat_df['smoker'] = self.cat_df['smoker'].map(
                {'yes': 1, 'no': 0})
            # self.logger_object.log(self.file_object, f'data col value:{self.cat_df.head(5)}')


            # Using the dummy encoding to encode the categorical columns to numerical ones


            self.cat_df = pd.get_dummies(self.cat_df, drop_first=True)
            # self.logger_object.log(self.file_object, f'data col value:{self.cat_df.head(5)}')


            self.data.drop(columns=self.data.select_dtypes(include=['object']).columns, inplace=True)
            # self.logger_object.log(self.file_object, f'data col value:{self.cat_df.head(5)}')
            self.data= pd.concat([self.cat_df,self.data],axis=1)
            self.logger_object.log(self.file_object, 'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

#
# log_writer = logger.App_Logger()
# file_object = open("../Prediction_Logs/Prediction_Log.txt", 'a+')
# data=pd.read_csv("G:\\backUp\\PycharmProjects\\InsurancePremium\\test_file\\ayash.csv")
#
# pre=Preprocessing(log_writer,file_object)
#
# print(data)
# test=pre.checking_missing_values(data)
#
# print(test)






        
















