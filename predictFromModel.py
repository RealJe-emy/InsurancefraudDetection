import traceback
import pandas as pd
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import os


class prediction:

    def __init__(self, path):
        self.file_object = "Prediction_Logs/Prediction_Log.txt"
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)
        # Define the expected column order from training
        self.expected_columns = [
            'months_as_customer', 'policy_deductable', 'policy_annual_premium',
            'incident_severity', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
            'bodily_injuries', 'property_damage', 'police_report_available',
            # Add any other columns that remain after dropping
        ]

    def predictionFromModel(self):
        try:
            self.log_writer.log(self.file_object, '=== Starting Prediction Process ===')

            # 1. Data Loading
            self.log_writer.log(self.file_object, 'Loading data...')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            data = data_getter.get_data()
            self.log_writer.log(self.file_object, f'Initial data shape: {data.shape}')

            # 2. Column Removal
            columns_to_drop = [
                'policy_number', 'policy_bind_date', 'policy_state', 'insured_zip',
                'incident_location', 'incident_date', 'incident_state', 'incident_city',
                'insured_hobbies', 'auto_make', 'auto_model', 'auto_year', 'age',
                'total_claim_amount'
            ]
            self.log_writer.log(self.file_object, f'Dropping columns: {columns_to_drop}')
            data = data.drop(columns=columns_to_drop, errors='ignore')

            # 3. Column Order Validation
            missing_cols = [col for col in self.expected_columns if col not in data.columns]
            if missing_cols:
                error_msg = f'Missing expected columns: {missing_cols}'
                self.log_writer.log(self.file_object, error_msg)
                raise ValueError(error_msg)

            data = data[self.expected_columns]
            self.log_writer.log(self.file_object, f'Final data columns: {list(data.columns)}')

            # 4. Preprocessing
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # Handle missing values
            data.replace('?', np.NaN, inplace=True)
            is_null_present, cols_with_missing = preprocessor.is_null_present(data)
            if is_null_present:
                self.log_writer.log(self.file_object, f'Imputing missing values in: {cols_with_missing}')
                data = preprocessor.impute_missing_values(data, cols_with_missing)

            # Encode categorical columns
            self.log_writer.log(self.file_object, 'Encoding categorical columns...')
            data = preprocessor.encode_categorical_columns(data)

            # Scale numerical columns
            self.log_writer.log(self.file_object, 'Scaling numerical columns...')
            data = preprocessor.scale_numerical_columns(data)

            # 5. Model Prediction
            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)

            # Load KMeans model
            self.log_writer.log(self.file_object, 'Loading KMeans model...')
            kmeans = file_loader.load_model('KMeans')

            # Cluster prediction
            self.log_writer.log(self.file_object, 'Predicting clusters...')
            clusters = kmeans.predict(data)
            data['clusters'] = clusters
            unique_clusters = np.unique(clusters)
            self.log_writer.log(self.file_object, f'Found clusters: {unique_clusters}')

            predictions = []
            for cluster_num in unique_clusters:
                self.log_writer.log(self.file_object, f'Processing cluster {cluster_num}...')

                # Get cluster-specific data
                cluster_data = data[data['clusters'] == cluster_num].drop(['clusters'], axis=1)

                # Load cluster-specific model
                model_name = file_loader.find_correct_model_file(cluster_num)
                self.log_writer.log(self.file_object, f'Loading model: {model_name}')
                model = file_loader.load_model(model_name)

                # Make predictions
                cluster_preds = model.predict(cluster_data)
                predictions.extend(['Y' if pred == 1 else 'N' for pred in cluster_preds])

            # 6. Save Results
            self.log_writer.log(self.file_object, 'Saving prediction results...')
            final = pd.DataFrame({'Predictions': predictions})
            path = "Prediction_Output_File/Predictions.csv"
            final.to_csv(path, index=False)

            self.log_writer.log(self.file_object, '=== Prediction Completed Successfully ===')
            return path

        except Exception as ex:
            # Get the root cause if this is a nested exception
            root_error = str(ex)
            while hasattr(ex, '__cause__') and ex.__cause__:
                root_error = str(ex.__cause__)
                ex = ex.__cause__

            error_msg = f'Prediction failed: {root_error}'
            self.log_writer.log(self.file_object, error_msg)
            self.log_writer.log(self.file_object, "Full error traceback:\n" + traceback.format_exc())

            # Raise a clean error without nesting
            raise Exception(error_msg) from None