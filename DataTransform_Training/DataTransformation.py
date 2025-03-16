from application_logging.logger import App_Logger
from os import listdir
import pandas

class dataTransform:
    """
    This class is used for transforming the Good Raw Training Data before loading it into the database.
    """

    def __init__(self):
        self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()

    def replaceMissingWithNull(self):
        """
        Method Name: replaceMissingWithNull
        Description: This method replaces missing values in columns with "NULL" to store in the table.
        Output: None
        On Failure: Raises Exception
        """
        log_file = open("Training_Logs/dataTransformLog.txt", "a+")
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                data = pandas.read_csv(self.goodDataPath + "/" + file)
                # List of columns with string datatype variables
                columns = [
                    "policy_bind_date", "policy_state", "policy_csl", "insured_sex",
                    "insured_education_level", "insured_occupation", "insured_hobbies",
                    "insured_relationship", "incident_state", "incident_date",
                    "incident_type", "collision_type", "incident_severity",
                    "authorities_contacted", "incident_city", "incident_location",
                    "property_damage", "police_report_available", "auto_make",
                    "auto_model", "fraud_reported"
                ]

                for col in columns:
                    data[col] = data[col].apply(lambda x: "'" + str(x) + "'")

                data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.logger.log(log_file, f"{file}: Quotes added successfully!!")
        except Exception as e:
            self.logger.log(log_file, f"Data Transformation failed: {str(e)}")
            raise e
        finally:
            log_file.close()