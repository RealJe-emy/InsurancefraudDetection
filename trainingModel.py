from datetime import datetime
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import DataTransform
from application_logging import logger
import os

class TrainValidation:
    def __init__(self, path, db_name='Training', log_file="Training_Logs/Training_Main_Log.txt"):
        """
        Initialize the TrainValidation class.

        Parameters:
        - path (str): Path to the raw data folder.
        - db_name (str): Name of the database to use. Default is 'Training'.
        - log_file (str): Path to the log file. Default is "Training_Logs/Training_Main_Log.txt".
        """
        self.raw_data = Raw_Data_validation(path)
        self.data_transform = DataTransform(data_type="training")
        self.db_operation = dBOperation()
        self.db_name = db_name
        self.log_file = log_file
        self.log_writer = logger.App_Logger()

    def train_validation(self):
        """
        Perform validation, transformation, and insertion of training data.
        """
        try:
            self.log_writer.log(self.log_file, 'Start of Validation on files for Training!!')

            # Extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()

            # Validating file names
            regex = self.raw_data.manualRegexCreation()
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

            # Validating column length
            self.raw_data.validateColumnLength(noofcolumns)

            # Validating missing values in columns
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.log_file, "Raw Data Validation Complete!!")

            # Data Transformation
            self.log_writer.log(self.log_file, "Starting Data Transformation!!")
            self.data_transform.replace_missing_with_null()
            self.log_writer.log(self.log_file, "Data Transformation Completed!!!")

            # Database Operations
            self.log_writer.log(self.log_file, "Creating Training_Database and tables on the basis of given schema!!!")
            self.db_operation.createTableDb(self.db_name, column_names)
            self.log_writer.log(self.log_file, "Table creation Completed!!")

            self.log_writer.log(self.log_file, "Insertion of Data into Table started!!!!")
            self.db_operation.insertIntoTableGoodData(self.db_name)
            self.log_writer.log(self.log_file, "Insertion in Table completed!!!")

            # Cleanup
            self.log_writer.log(self.log_file, "Deleting Good Data Folder!!!")
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.log_file, "Good_Data folder deleted!!!")

            self.log_writer.log(self.log_file, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.log_file, "Bad files moved to archive!! Bad folder Deleted!!")

            self.log_writer.log(self.log_file, "Validation Operation completed!!")
            self.log_writer.log(self.log_file, "Extracting csv file from table")
            self.db_operation.selectingDatafromtableintocsv(self.db_name)

        except Exception as e:
            self.log_writer.log(self.log_file, f"Error during training validation: {str(e)}")
            raise e
        finally:
            if hasattr(self, 'file_object'):
                self.file_object.close()