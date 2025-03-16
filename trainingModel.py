"""
This is the Entry point for Training the Machine Learning Model.
"""

# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import numpy as np
import pandas as pd

# Creating the common Logging object
class trainModel:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Step 1: Getting the data from the source
            self.log_writer.log(self.file_object, 'Loading data from the source...')
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()
            self.log_writer.log(self.file_object, 'Data loaded successfully.')

            # Step 2: Data Preprocessing
            self.log_writer.log(self.file_object, 'Starting data preprocessing...')
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # Remove unnecessary columns
            columns_to_remove = [
                'policy_number', 'policy_bind_date', 'policy_state', 'insured_zip',
                'incident_location', 'incident_date', 'incident_state', 'incident_city',
                'insured_hobbies', 'auto_make', 'auto_model', 'auto_year', 'age',
                'total_claim_amount'
            ]
            data = preprocessor.remove_columns(data, columns_to_remove)
            self.log_writer.log(self.file_object, 'Unnecessary columns removed.')

            # Replace '?' with NaN
            data.replace('?', np.NaN, inplace=True)
            self.log_writer.log(self.file_object, "Replaced '?' with NaN.")

            # Check for missing values
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
            if is_null_present:
                self.log_writer.log(self.file_object, 'Missing values found. Imputing missing values...')
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)
                self.log_writer.log(self.file_object, 'Missing values imputed.')
            else:
                self.log_writer.log(self.file_object, 'No missing values found.')

            # Encode categorical data
            self.log_writer.log(self.file_object, 'Encoding categorical data...')
            data = preprocessor.encode_categorical_columns(data)
            self.log_writer.log(self.file_object, 'Categorical data encoded.')

            # Separate features and labels
            self.log_writer.log(self.file_object, 'Separating features and labels...')
            X, Y = preprocessor.separate_label_feature(data, label_column_name='fraud_reported')
            self.log_writer.log(self.file_object, 'Features and labels separated.')

            # Step 3: Applying the clustering approach
            self.log_writer.log(self.file_object, 'Applying clustering approach...')
            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer)  # object initialization.
            number_of_clusters = kmeans.elbow_plot(X)  # using the elbow plot to find the number of optimum clusters
            self.log_writer.log(self.file_object, f'Optimal number of clusters found: {number_of_clusters}')

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)
            self.log_writer.log(self.file_object, 'Data divided into clusters.')

            # Create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y
            self.log_writer.log(self.file_object, 'Cluster labels added to the dataset.')

            # Get the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()
            self.log_writer.log(self.file_object, f'Unique clusters: {list_of_clusters}')

            # Step 4: Parsing all the clusters and looking for the best ML algorithm to fit on individual cluster
            for i in list_of_clusters:
                self.log_writer.log(self.file_object, f'Processing cluster {i}...')
                cluster_data = X[X['Cluster'] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # Splitting the data into training and test set for each cluster
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1/3, random_state=355)
                self.log_writer.log(self.file_object, f'Data split into training and test sets for cluster {i}.')

                # Proceeding with more data pre-processing steps
                x_train = preprocessor.scale_numerical_columns(x_train)
                x_test = preprocessor.scale_numerical_columns(x_test)
                self.log_writer.log(self.file_object, f'Data scaled for cluster {i}.')

                # Finding the best model for the cluster
                model_finder = tuner.Model_Finder(self.file_object, self.log_writer)  # object initialization
                best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)
                self.log_writer.log(self.file_object, f'Best model found for cluster {i}: {best_model_name}')

                # Saving the best model to the directory
                file_op = file_methods.File_Operation(self.file_object, self.log_writer)
                save_model = file_op.save_model(best_model, best_model_name + str(i))
                self.log_writer.log(self.file_object, f'Model saved for cluster {i}.')

            # Logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # Logging the unsuccessful Training
            self.log_writer.log(self.file_object, f'Unsuccessful End of Training: {str(e)}')
            self.file_object.close()
            raise e