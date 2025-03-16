from datetime import datetime
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTransform  # Correct import
from application_logging import logger
from trainingModel import trainModel  # Import the trainingModel class


class train_validation:
    def __init__(self, path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()  # Initialize the dataTransform class
        self.dBOperation = dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def train_validation(self):
        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files for Training!!')

            # Step 1: Extract schema details
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            self.log_writer.log(self.file_object, 'Schema details extracted successfully.')

            # Step 2: Validate file names
            regex = self.raw_data.manualRegexCreation()
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            self.log_writer.log(self.file_object, 'File name validation completed.')

            # Step 3: Validate column length
            self.raw_data.validateColumnLength(noofcolumns)
            self.log_writer.log(self.file_object, 'Column length validation completed.')

            # Step 4: Validate missing values
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, 'Missing values validation completed.')

            # Step 5: Data Transformation
            self.log_writer.log(self.file_object, 'Starting Data Transformation!!')
            self.dataTransform.replaceMissingWithNull()
            self.log_writer.log(self.file_object, 'Data Transformation Completed!!!')

            # Step 6: Create database and tables
            self.log_writer.log(self.file_object,
                                'Creating Training_Database and tables on the basis of given schema!!!')
            self.dBOperation.createTableDb('Training', column_names)
            self.log_writer.log(self.file_object, 'Table creation Completed!!')

            # Step 7: Insert data into the table
            self.log_writer.log(self.file_object, 'Insertion of Data into Table started!!!!')
            self.dBOperation.insertIntoTableGoodData('Training')
            self.log_writer.log(self.file_object, 'Insertion in Table completed!!!')

            # Step 8: Delete the Good Data folder
            self.log_writer.log(self.file_object, 'Deleting Good Data Folder!!!')
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, 'Good_Data folder deleted!!!')

            # Step 9: Move bad files to archive and delete Bad_Data folder
            self.log_writer.log(self.file_object, 'Moving bad files to Archive and deleting Bad_Data folder!!!')
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_object, 'Bad files moved to archive!! Bad folder Deleted!!')

            # Step 10: Export data from table to CSV
            self.log_writer.log(self.file_object, 'Extracting csv file from table')
            self.dBOperation.selectingDatafromtableintocsv('Training')
            self.log_writer.log(self.file_object, 'Validation Operation completed!!')

            # Step 11: Invoke the training process
            self.log_writer.log(self.file_object, 'Starting the training process...')
            trainer = trainModel()  # Initialize the trainingModel class
            trainer.trainingModel()  # Start the training process
            self.log_writer.log(self.file_object, 'Training process completed successfully!!')

            self.file_object.close()

        except Exception as e:
            self.log_writer.log(self.file_object, f'Error during validation or training: {str(e)}')
            self.file_object.close()
            raise e