# import shutil
# import sqlite3
# from datetime import datetime
# from os import listdir
# import os
# import csv
# from application_logging.logger import App_Logger
#
#
# class dBOperation:
#     """
#       This class shall be used for handling all the SQL operations.
#
#
#
#       """
#     def __init__(self):
#         self.path = 'Training_Database/'
#         self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
#         self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
#         self.logger = App_Logger()
#
#
#     def dataBaseConnection(self,DatabaseName):
#
#         """
#                 Method Name: dataBaseConnection
#                 Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
#                 Output: Connection to the DB
#                 On Failure: Raise ConnectionError
#
#
#
#                 """
#         if not DatabaseName:
#             raise ValueError("DatabaseName cannot be empty or None!")
#
#         print(f"Connecting to database: {DatabaseName}")
#
#
#         try:
#             conn = sqlite3.connect(self.path+DatabaseName+'.db')
#
#             file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
#             self.logger.log(file, "Opened %s database successfully" % DatabaseName)
#             file.close()
#         except ConnectionError:
#             file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
#             self.logger.log(file, "Error while connecting to database: %s" %ConnectionError)
#             file.close()
#             raise ConnectionError
#         return conn
#
#     def createTableDb(self,DatabaseName,column_names):
#         """
#                         Method Name: createTableDb
#                         Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
#                         Output: None
#                         On Failure: Raise Exception
#
#
#
#                         """
#         conn = None
#         try:
#             conn = self.dataBaseConnection(DatabaseName)
#             c=conn.cursor()
#             c.execute("SELECT count(name)  FROM sqlite_master WHERE type = 'table'AND name = 'Good_Raw_Data'")
#             if c.fetchone()[0] ==1:
#                 conn.close()
#                 file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
#                 self.logger.log(file, "Tables created successfully!!")
#                 file.close()
#
#                 file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
#                 self.logger.log(file, "Closed %s database successfully" % DatabaseName)
#                 file.close()
#
#             else:
#
#                 for key in column_names.keys():
#                     type = column_names[key]
#
#                     #in try block we check if the table exists, if yes then add columns to the table
#                     # else in catch block we will create the table
#                     try:
#                         #cur = cur.execute("SELECT name FROM {dbName} WHERE type='table' AND name='Good_Raw_Data'".format(dbName=DatabaseName))
#                         conn.execute('ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,dataType=type))
#                     except:
#                         conn.execute('CREATE TABLE  Good_Raw_Data ({column_name} {dataType})'.format(column_name=key, dataType=type))
#
#
#                     # try:
#                     #     #cur.execute("SELECT name FROM {dbName} WHERE type='table' AND name='Bad_Raw_Data'".format(dbName=DatabaseName))
#                     #     conn.execute('ALTER TABLE Bad_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,dataType=type))
#                     #
#                     # except:
#                     #     conn.execute('CREATE TABLE Bad_Raw_Data ({column_name} {dataType})'.format(column_name=key, dataType=type))
#
#
#                 conn.close()
#
#                 file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
#                 self.logger.log(file, "Tables created successfully!!")
#                 file.close()
#
#                 file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
#                 self.logger.log(file, "Closed %s database successfully" % DatabaseName)
#                 file.close()
#
#         except Exception as e:
#             file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
#             self.logger.log(file, "Error while creating table: %s " % e)
#             file.close()
#             conn.close()
#             file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
#             self.logger.log(file, "Closed %s database successfully" % DatabaseName)
#             file.close()
#             raise e
#
#
#     def insertIntoTableGoodData(self,Database):
#
#         """
#                                Method Name: insertIntoTableGoodData
#                                Description: This method inserts the Good data files from the Good_Raw folder into the
#                                             above created table.
#                                Output: None
#                                On Failure: Raise Exception
#
#
#
#         """
#
#         conn = self.dataBaseConnection(Database)
#         goodFilePath= self.goodFilePath
#         badFilePath = self.badFilePath
#         onlyfiles = [f for f in listdir(goodFilePath)]
#         log_file = open("Training_Logs/DbInsertLog.txt", 'a+')
#
#         for file in onlyfiles:
#             try:
#                 with open(goodFilePath+'/'+file, "r") as f:
#                     next(f)
#                     reader = csv.reader(f, delimiter="\n")
#                     for line in enumerate(reader):
#                         for list_ in (line[1]):
#                             try:
#                                 conn.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=(list_)))
#                                 self.logger.log(log_file," %s: File loaded successfully!!" % file)
#                                 conn.commit()
#                             except Exception as e:
#                                 raise e
#
#             except Exception as e:
#
#                 conn.rollback()
#                 self.logger.log(log_file,"Error while creating table: %s " % e)
#                 shutil.move(goodFilePath+'/' + file, badFilePath)
#                 self.logger.log(log_file, "File Moved Successfully %s" % file)
#                 log_file.close()
#                 conn.close()
#
#         conn.close()
#         log_file.close()
#
#
#     def selectingDatafromtableintocsv(self,Database):
#
#         """
#                                Method Name: selectingDatafromtableintocsv
#                                Description: This method exports the data in GoodData table as a CSV file. in a given location.
#                                             above created .
#                                Output: None
#                                On Failure: Raise Exception
#
#
#
#         """
#
#         self.fileFromDb = 'Training_FileFromDB/'
#         self.fileName = 'InputFile.csv'
#         log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
#         try:
#             conn = self.dataBaseConnection(Database)
#             sqlSelect = "SELECT *  FROM Good_Raw_Data"
#             cursor = conn.cursor()
#
#             cursor.execute(sqlSelect)
#
#             results = cursor.fetchall()
#             # Get the headers of the csv file
#             headers = [i[0] for i in cursor.description]
#
#             #Make the CSV ouput directory
#             if not os.path.isdir(self.fileFromDb):
#                 os.makedirs(self.fileFromDb)
#
#             # Open CSV file for writing.
#             csvFile = csv.writer(open(self.fileFromDb + self.fileName, 'w', newline=''),delimiter=',', lineterminator='\r\n',quoting=csv.QUOTE_ALL, escapechar='\\')
#
#             # Add the headers and data to the CSV file.
#             csvFile.writerow(headers)
#             csvFile.writerows(results)
#
#             self.logger.log(log_file, "File exported successfully!!!")
#             log_file.close()
#
#         except Exception as e:
#             self.logger.log(log_file, "File exporting failed. Error : %s" %e)
#             log_file.close()
#
import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from application_logging.logger import App_Logger


class dBOperation:
    """
      This class shall be used for handling all the SQL operations.
    """
    def __init__(self):
        self.path = 'Training_Database/'
        self.database_name = 'TrainingDB'  # Hardcoded database name
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()

    def dataBaseConnection(self):
        """
        Method Name: dataBaseConnection
        Description: This method creates the database with the given name and if it already exists, it opens the connection.
        Output: Connection to the DB
        On Failure: Raise ConnectionError
        """
        try:
            conn = sqlite3.connect(self.path + self.database_name + '.db')
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, f"Opened {self.database_name} database successfully")
            file.close()
            return conn
        except ConnectionError as e:
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, f"Error while connecting to database: {str(e)}")
            file.close()
            raise e

    def createTableDb(self, column_names):
        """
        Method Name: createTableDb
        Description: This method creates a table in the database for inserting Good data after validation.
        Output: None
        On Failure: Raise Exception
        """
        try:
            conn = self.dataBaseConnection()
            c = conn.cursor()
            c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Good_Raw_Data'")
            if c.fetchone()[0] == 1:
                file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Table already exists!")
                file.close()
            else:
                columns = ", ".join([f'"{key}" {value}' for key, value in column_names.items()])
                conn.execute(f'CREATE TABLE Good_Raw_Data ({columns})')
                file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Table created successfully!")
                file.close()
            conn.close()
        except Exception as e:
            file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, f"Error while creating table: {str(e)}")
            file.close()
            raise e

    def insertIntoTableGoodData(self):
        """
        Method Name: insertIntoTableGoodData
        Description: Inserts Good data files into the database.
        Output: None
        On Failure: Raise Exception
        """
        conn = self.dataBaseConnection()
        log_file = open("Training_Logs/DbInsertLog.txt", 'a+')
        onlyfiles = [f for f in listdir(self.goodFilePath)]

        for file in onlyfiles:
            try:
                with open(self.goodFilePath + '/' + file, "r") as f:
                    next(f)  # Skip header
                    reader = csv.reader(f, delimiter="\n")
                    for line in reader:
                        for list_ in line:
                            try:
                                conn.execute(f'INSERT INTO Good_Raw_Data VALUES ({list_})')
                                conn.commit()
                            except Exception as e:
                                raise e
                self.logger.log(log_file, f"{file}: File loaded successfully!")
            except Exception as e:
                conn.rollback()
                self.logger.log(log_file, f"Error inserting data: {str(e)}")
                shutil.move(self.goodFilePath + '/' + file, self.badFilePath)
                self.logger.log(log_file, f"File moved to bad folder: {file}")
        conn.close()
        log_file.close()

    def selectingDatafromtableintocsv(self):
        """
        Method Name: selectingDatafromtableintocsv
        Description: Exports data from the database table to a CSV file.
        Output: None
        On Failure: Raise Exception
        """
        fileFromDb = 'Training_FileFromDB/'
        fileName = 'InputFile.csv'
        log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
        try:
            conn = self.dataBaseConnection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Good_Raw_Data")
            results = cursor.fetchall()
            headers = [i[0] for i in cursor.description]

            if not os.path.isdir(fileFromDb):
                os.makedirs(fileFromDb)

            with open(fileFromDb + fileName, 'w', newline='') as f:
                csvFile = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
                csvFile.writerow(headers)
                csvFile.writerows(results)

            self.logger.log(log_file, "File exported successfully!")
        except Exception as e:
            self.logger.log(log_file, f"File exporting failed: {str(e)}")
        finally:
            log_file.close()

