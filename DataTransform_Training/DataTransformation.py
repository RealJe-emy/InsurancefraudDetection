import os
import pandas as pd
from application_logging.logger import App_Logger


class DataTransform:
    """
     This class is responsible for transforming the Good Raw Data (both training and prediction)
     before loading it into the database.
     """

    def __init__(self, data_type="training"):
        """
          Initialize the DataTransform class.

          Parameters:
          - data_type (str): Type of data to transform. Can be "training" or "prediction".
          """
        if data_type == "training":
            self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
            self.log_file = "Training_Logs/dataTransformLog.txt"
        else:
            self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
            self.log_file = "Prediction_Logs/dataTransformLog.txt"

        self.logger = App_Logger()

    def replace_missing_with_null(self):
        """
          Replaces missing values in string columns with "NULL" for database insertion.
          """
        log_file = open(self.log_file, 'a+')
        try:
            onlyfiles = [f for f in os.listdir(self.goodDataPath)]
            for file in onlyfiles:
                file_path = os.path.join(self.goodDataPath, file)
                data = pd.read_csv(file_path)

                # List of columns with string datatype
                string_columns = [
                    "policy_bind_date", "policy_state", "policy_csl", "insured_sex",
                    "insured_education_level", "insured_occupation", "insured_hobbies",
                    "insured_relationship", "incident_state", "incident_date", "incident_type",
                    "collision_type", "incident_severity", "authorities_contacted", "incident_city",
                    "incident_location", "property_damage", "police_report_available", "auto_make",
                    "auto_model", "fraud_reported"
                ]

                # Replace missing values with "NULL"
                for col in string_columns:
                    data[col] = data[col].apply(lambda x: f"'{x}'" if pd.notnull(x) else "'NULL'")

                # Save the transformed data
                data.to_csv(file_path, index=None, header=True)
                self.logger.log(log_file, f"{file}: Quotes added successfully!!")
        except Exception as e:
            self.logger.log(log_file, f"Data Transformation failed because:: {e}")
            raise e
        finally:
            log_file.close()