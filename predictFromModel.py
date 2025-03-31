import traceback
import pandas as pd
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import os
import json


class prediction:

    def __init__(self, path):
        self.file_object = "Prediction_Logs/Prediction_Log.txt"
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

        # Define the expected column order from training
        self.expected_columns = [
            'months_as_customer', 'policy_deductable', 'policy_annual_premium',
            'incident_severity', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
            'bodily_injuries', 'property_damage', 'police_report_available'
        ]

        # Configuration paths
        self.model_dir = "models/"
        self.output_dir = "Prediction_Output_File/"

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def _log_data_sample(self, data, stage):
        """Log a sample of the data at different stages for debugging"""
        sample = data.head(1).to_dict(orient='records')[0]
        self.log_writer.log(self.file_object,
                            f"{stage} data sample:\n{json.dumps(sample, indent=2)}")

    def predictionFromModel(self):
        try:
            self.log_writer.log(self.file_object, '=== Starting Prediction Process ===')
            self.log_writer.log(self.file_object, f"Current working directory: {os.getcwd()}")
            self.log_writer.log(self.file_object, f"Model directory contents: {os.listdir(self.model_dir)}")

            # 1. Data Loading
            self.log_writer.log(self.file_object, 'Loading data...')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            data = data_getter.get_data()
            self.log_writer.log(self.file_object, f'Initial data shape: {data.shape}')
            self._log_data_sample(data, "Raw input")

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
                error_msg = f'Missing expected columns: {missing_cols}\nExisting columns: {data.columns.tolist()}'
                self.log_writer.log(self.file_object, error_msg)
                raise ValueError(error_msg)

            data = data[self.expected_columns]
            self.log_writer.log(self.file_object, f'Final data columns: {list(data.columns)}')
            self._log_data_sample(data, "After column selection")

            # 4. Preprocessing
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # Handle missing values
            data.replace('?', np.NaN, inplace=True)
            is_null_present, cols_with_missing = preprocessor.is_null_present(data)
            if is_null_present:
                self.log_writer.log(self.file_object, f'Imputing missing values in: {cols_with_missing}')
                data = preprocessor.impute_missing_values(data, cols_with_missing)
                self._log_data_sample(data, "After imputation")

            # Encode categorical columns
            self.log_writer.log(self.file_object, 'Encoding categorical columns...')
            data = preprocessor.encode_categorical_columns(data)
            self._log_data_sample(data, "After encoding")

            # Scale numerical columns
            self.log_writer.log(self.file_object, 'Scaling numerical columns...')
            data = preprocessor.scale_numerical_columns(data)
            self._log_data_sample(data, "After scaling")

            # 5. Model Prediction
            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)

            # Load KMeans model with enhanced error handling
            try:
                self.log_writer.log(self.file_object, 'Loading KMeans model...')
                kmeans = file_loader.load_model('KMeans')
                self.log_writer.log(self.file_object, 'KMeans model loaded successfully')
            except Exception as e:
                model_files = [f for f in os.listdir(self.model_dir) if 'KMeans' in f]
                self.log_writer.log(self.file_object,
                                    f"Available model files: {model_files}\nError loading KMeans: {str(e)}")
                raise Exception(f"KMeans model loading failed. Available models: {model_files}. Error: {str(e)}")

            # Cluster prediction with validation
            try:
                self.log_writer.log(self.file_object, 'Predicting clusters...')
                clusters = kmeans.predict(data)
                if len(clusters) == 0:
                    raise Exception("Cluster prediction returned empty results")
                data['clusters'] = clusters
                unique_clusters = np.unique(clusters)
                self.log_writer.log(self.file_object, f'Found clusters: {unique_clusters}')
            except Exception as e:
                self.log_writer.log(self.file_object,
                                    f'Cluster prediction failed. Data shape: {data.shape}. Error: {str(e)}')
                raise Exception(f"Cluster prediction failed: {str(e)}")

            predictions = []
            failed_clusters = []

            for cluster_num in unique_clusters:
                try:
                    self.log_writer.log(self.file_object, f'Processing cluster {cluster_num}...')
                    cluster_data = data[data['clusters'] == cluster_num].drop(['clusters'], axis=1)

                    if cluster_data.empty:
                        self.log_writer.log(self.file_object, f'No data points in cluster {cluster_num}')
                        continue

                    # Load cluster-specific model
                    try:
                        model_name = file_loader.find_correct_model_file(cluster_num)
                        self.log_writer.log(self.file_object, f'Loading model: {model_name}')
                        model = file_loader.load_model(model_name)
                    except Exception as e:
                        available_models = [f for f in os.listdir(self.model_dir) if str(cluster_num) in f]
                        self.log_writer.log(self.file_object,
                                            f"Available models for cluster {cluster_num}: {available_models}\nError: {str(e)}")
                        raise Exception(f"Model for cluster {cluster_num} not found. Available: {available_models}")

                    # Make predictions
                    try:
                        cluster_preds = model.predict(cluster_data)
                        predictions.extend(['Y' if pred == 1 else 'N' for pred in cluster_preds])
                        self.log_writer.log(self.file_object,
                                            f'Cluster {cluster_num} predictions: {len(cluster_preds)} records')
                    except Exception as e:
                        self.log_writer.log(self.file_object,
                                            f'Prediction failed for cluster {cluster_num}. Data shape: {cluster_data.shape}. Error: {str(e)}')
                        raise Exception(f"Prediction failed for cluster {cluster_num}: {str(e)}")

                except Exception as e:
                    failed_clusters.append(str(cluster_num))
                    self.log_writer.log(self.file_object,
                                        f'Cluster {cluster_num} processing failed. Error: {str(e)}')
                    continue  # Continue with other clusters if one fails

            if not predictions:
                raise Exception(f"No predictions generated. Failed clusters: {', '.join(failed_clusters)}")

            # 6. Save Results
            self.log_writer.log(self.file_object, 'Saving prediction results...')
            final = pd.DataFrame({'Predictions': predictions})
            path = os.path.join(self.output_dir, "Predictions.csv")
            final.to_csv(path, index=False)

            self.log_writer.log(self.file_object, f'Predictions saved to {path}')
            self.log_writer.log(self.file_object, '=== Prediction Completed Successfully ===')

            return path

        except Exception as ex:
            # Get detailed error information
            error_type = type(ex).__name__
            tb = traceback.format_exc()

            # Log detailed error information
            self.log_writer.log(self.file_object, f'ERROR TYPE: {error_type}')
            self.log_writer.log(self.file_object, f'ERROR MESSAGE: {str(ex)}')
            self.log_writer.log(self.file_object, f'TRACEBACK:\n{tb}')

            # Create a more informative error message
            error_msg = (f"Prediction failed at stage: {error_type}\n"
                         f"Details: {str(ex)}\n"
                         f"Check logs for complete traceback")

            # Raise a clean error with all context
            raise Exception(error_msg) from None