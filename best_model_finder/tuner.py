
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from application_logging import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from Processing.data_preprocessing import Preprocessing
class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: Sambit Kumar BEhera
                Version: 1.0
                Revisions: None

                """

    def __init__(self):
        self.file_object = open("../ModelFinder_Logs/model_Log.txt", 'a+')
        self.logger_object = logger.App_Logger()
        self.rf_regressor=RandomForestRegressor()
        self.xgb_regressor = XGBRegressor()

    def get_best_params_for_randomforest(self,train_x,train_y):




        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_randomforest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.random_grid = {'bootstrap': [True, False],
                           'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90],
                           'max_features': ['auto', 'sqrt'],
                           'min_samples_leaf': [1, 2, 4],
                           'min_samples_split': [2, 5, 10],
                           'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator =self.rf_regressor, param_grid = self.random_grid, cv = 3, verbose=2, n_jobs = -1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.bootstrap = self.grid.best_params_['bootstrap']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']



            #creating a new model with the best parameters
            self.rf_regressor = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,
                                                      bootstrap=self.bootstrap)

            # training the mew model
            self.rf_regressor.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_RandomForest method of the Model_Finder class')

            return self.rf_regressor
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_RandomForest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Ramdom Forest training  failed. Exited the get_best_params_for_RandomForest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Sambit kumarBEhera
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {
                                        'n_estimators':[100, 500, 900, 1100, 1500,1800,2000],
                'max_depth' : [2, 3, 5, 10, 15,20,25,30],
                'booster':['gbtree','gblinear'],
                'learning_rate':[0.05,0.1,0.15,0.20,0.25,0.30,0.35],
                'min_child_weight':[1,2,3,4],
                'base_score':[0.25,0.5,0.75,1]
            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator =self.xgb_regressor,param_grid=self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.booster = self.grid.best_params_['booster']
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.min_child_weight = self.grid.best_params_['min_child_weight']
            self.base_score = self.grid.best_params_['base_score']

            # creating a new model with the best parameters
            self.xgb = self.xgb_regressor(booster=self.booster, max_depth=self.max_depth, n_estimators= self.n_estimators,learning_rate=self.learning_rate ,min_child_weight=self.min_child_weight,base_score=self.base_score,n_jobs=-1)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Sambit kumar BEhera
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model
            self.rf_reg = self.get_best_params_for_xgboost(train_x, train_y)

            self.xgboost_score=self.xgboost.score(test_x,test_y)
            self.logger_object.log(self.file_object,f'Score for XGBoost is: {self.xgboost_score}')
            self.rf_reg_score = self.rf_reg.score(test_x, test_y)
            self.logger_object.log(self.file_object, f'Score for RandomForest is: {self.xgboost_score}')




            #comparing the two models
            if(self.xgboost_score <  self.rf_reg_score):
                return 'RandomForest',self.rf_reg
            else:
                return 'XGBoost',self.self.xgboost

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()


data=pd.read_csv('../data/insurance.csv')
pre=Preprocessing()
data=pre.outier_treatment(data,columns=['bmi'])
data=pre.scaling_of_numcol(data)
data=pre.encode_categorical_col(data)
x=data.drop('expenses',axis=1)
y=data['expenses']

train_x,test_x,train_y,test_y=train_test_split(x,y, random_state=355 )

tun=Model_Finder()
# print(tun.get_best_params_for_randomforest(train_x,train_y))
print(tun.get_best_params_for_xgboost(train_x,train_y))



