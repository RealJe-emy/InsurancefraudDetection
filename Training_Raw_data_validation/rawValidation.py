import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from application_logging.logger import App_Logger

class Raw_Data_validation:
    def __init__(self, path, log_file):
        """
        Method Name: __init__
        Description: Initializes the Raw_Data_validation object.
        Input:
            - path: Path to the directory containing the raw data files.
            - log_file: File object for logging.
        """
        self.Batch_Directory = path
        self.schema_path = 'schema_training.json'
        self.logger = App_Logger()
        self.log_file = log_file  # Use the provided log_file for logging

    def valuesFromSchema(self):
        """
        Method Name: valuesFromSchema
        Description: Extracts schema details for validation.
        Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns
        On Failure: Raises Exception
        """
        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']

            # Log the schema details
            self.logger.log(self.log_file, "Schema details extracted successfully.")
            return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns
        except Exception as e:
            self.logger.log(self.log_file, f"Error extracting schema details: {str(e)}")
            raise e

    def manualRegexCreation(self):
        """
        Method Name: manualRegexCreation
        Description: Creates a regex pattern for validating file names.
        Output: Regex pattern
        On Failure: None
        """
        regex = "['fraudDetection']+['\_'']+[\d_]+[\d]+\.csv"
        self.logger.log(self.log_file, "Regex pattern created successfully.")
        return regex

    def validationFileNameRaw(self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile):
        """
        Method Name: validationFileNameRaw
        Description: Validates file names based on the regex pattern.
        Output: None
        On Failure: Raises Exception
        """
        try:
            # Delete the directories for good and bad data if they exist
            self.deleteExistingBadDataTrainingFolder()
            self.deleteExistingGoodDataTrainingFolder()
            self.createDirectoryForGoodBadRawData()

            # Get list of files in the batch directory
            onlyfiles = [f for f in listdir(self.Batch_Directory)]

            # Validate file names
            for filename in onlyfiles:
                if re.match(regex, filename):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = re.split('_', splitAtDot[0])
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy(os.path.join(self.Batch_Directory, filename), "Training_Raw_files_validated/Good_Raw")
                            self.logger.log(self.log_file, f"Valid File name: {filename}. File moved to Good_Raw folder.")
                        else:
                            shutil.copy(os.path.join(self.Batch_Directory, filename), "Training_Raw_files_validated/Bad_Raw")
                            self.logger.log(self.log_file, f"Invalid File Name: {filename}. File moved to Bad_Raw folder.")
                    else:
                        shutil.copy(os.path.join(self.Batch_Directory, filename), "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(self.log_file, f"Invalid File Name: {filename}. File moved to Bad_Raw folder.")
                else:
                    shutil.copy(os.path.join(self.Batch_Directory, filename), "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(self.log_file, f"Invalid File Name: {filename}. File moved to Bad_Raw folder.")

            self.logger.log(self.log_file, "File name validation completed.")
        except Exception as e:
            self.logger.log(self.log_file, f"File name validation failed: {str(e)}")
            raise e

    def validateColumnLength(self, NumberofColumns):
        """
        Method Name: validateColumnLength
        Description: Validates the number of columns in the file.
        Output: None
        On Failure: Raises Exception
        """
        try:
            self.logger.log(self.log_file, "Column Length Validation Started!!")
            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(self.log_file, f"Invalid Column Length for the file!! File moved to Bad Raw Folder :: {file}")
            self.logger.log(self.log_file, "Column Length Validation Completed!!")
        except Exception as e:
            self.logger.log(self.log_file, f"Error during column length validation: {str(e)}")
            raise e

    def validateMissingValuesInWholeColumn(self):
        """
        Method Name: validateMissingValuesInWholeColumn
        Description: Checks if any column has all missing values.
        Output: None
        On Failure: Raises Exception
        """
        try:
            self.logger.log(self.log_file, "Missing Values Validation Started!!")
            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(self.log_file, f"Invalid Column for the file!! File moved to Bad Raw Folder :: {file}")
                        break
                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
            self.logger.log(self.log_file, "Missing Values Validation Completed!!")
        except Exception as e:
            self.logger.log(self.log_file, f"Error during missing values validation: {str(e)}")
            raise e

    def moveBadFilesToArchiveBad(self):
        """
        Method Name: moveBadFilesToArchiveBad
        Description: Moves bad files to the archive folder.
        Output: None
        On Failure: Raises Exception
        """
        try:
            now = datetime.now()
            date = now.date()
            time = now.strftime("%H%M%S")

            source = 'Training_Raw_files_validated/Bad_Raw/'
            if os.path.isdir(source):
                path = "TrainingArchiveBadData"
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = 'TrainingArchiveBadData/BadData_' + str(date) + "_" + str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                self.logger.log(self.log_file, "Bad files moved to archive")

                path = 'Training_Raw_files_validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.logger.log(self.log_file, "Bad Raw Data Folder Deleted successfully!!")
        except Exception as e:
            self.logger.log(self.log_file, f"Error while moving bad files to archive: {str(e)}")
            raise e

    def deleteExistingGoodDataTrainingFolder(self):
        """
        Method Name: deleteExistingGoodDataTrainingFolder
        Description: Deletes the Good Data folder after loading files into the database.
        Output: None
        On Failure: Raises Exception
        """
        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                self.logger.log(self.log_file, "Good Data folder deleted successfully.")
        except Exception as e:
            self.logger.log(self.log_file, f"Error deleting Good Data folder: {str(e)}")
            raise e

    def deleteExistingBadDataTrainingFolder(self):
        """
        Method Name: deleteExistingBadDataTrainingFolder
        Description: Deletes the Bad Data folder.
        Output: None
        On Failure: Raises Exception
        """
        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                self.logger.log(self.log_file, "Bad Data folder deleted successfully.")
        except Exception as e:
            self.logger.log(self.log_file, f"Error deleting Bad Data folder: {str(e)}")
            raise e

    def createDirectoryForGoodBadRawData(self):
        """
        Method Name: createDirectoryForGoodBadRawData
        Description: Creates directories for Good_Raw and Bad_Raw data.
        Output: None
        On Failure: Raises Exception
        """
        try:
            path = "Training_Raw_files_validated/"
            if not os.path.isdir(path + "Bad_Raw/"):
                os.makedirs(path + "Bad_Raw/")
            if not os.path.isdir(path + "Good_Raw/"):
                os.makedirs(path + "Good_Raw/")
            self.logger.log(self.log_file, "Directories for Good_Raw and Bad_Raw created successfully.")
        except Exception as e:
            self.logger.log(self.log_file, f"Error creating directories: {str(e)}")
            raise e