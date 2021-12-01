import pymongo
from pymongo import MongoClient
import pandas as pd
from application_logging import logger

class DBupload:

    def __init__(self,dbname=None,collectionName=None):

        self.dBName = dbname
        self.collectionName = collectionName

        self.client = MongoClient("localhost", 27017, maxPoolSize=50)

        self.DB = self.client[self.dBName]
        self.collection = self.DB[self.collectionName]
        self.log_writer = logger.App_Logger()
        self.file_object = open("../DB_Logs/Insert_Log.txt", 'a+')


    def insertCSV(self,path=None):
        """
                   :param path: Path os csv File
                   :return: None
                   """
        try:
            if  self.dBName in self.client.list_database_names():
                self.log_writer.log(self.file_object, 'DB Already Exists')
            elif self.collection in self.DB.list_collections():
                print('Collection Already Exist.')
            else:
                df = pd.read_csv(path)
                data = df.to_dict('records')

                self.collection.insert_many(data, ordered=False)
                self.log_writer.log(self.file_object, 'All Data HAs been Updated')


        except Exception as e:
            self.log_writer.log(self.file_object, 'All Data HAs been Updated')

            raise Exception









up=DBupload('InsData','RawData')
up.insertCSV('../data/insurance.csv')





