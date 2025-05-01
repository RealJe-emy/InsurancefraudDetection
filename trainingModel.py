# trainingModel.py
# """
# This is the Entry point for Training the Machine Learning Model.

# Written By: iNeuron Intelligence
# Version: 1.0
# Revisions: None

# """


# # Doing the necessary imports
# from sklearn.model_selection import train_test_split
# from data_ingestion import data_loader
# from data_preprocessing import preprocessing
# from data_preprocessing import clustering
# from best_model_finder import tuner
# from file_operations import file_methods
# from application_logging import logger
# import numpy as np
# import pandas as pd

# #Creating the common Logging object


# class trainModel:

#     def __init__(self):
#         self.log_writer = logger.App_Logger()
#         self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
#     def trainingModel(self):
#         # Logging the start of Training
#         self.log_writer.log(self.file_object, 'Start of Training')
#         try:
#             # Getting the data from the source
#             data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
#             data=data_getter.get_data()


#             """doing the data preprocessing"""

#             preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
#             data=preprocessor.remove_columns(data,['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','age','total_claim_amount']) # remove the column as it doesn't contribute to prediction.
#             data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation

#             # check if missing values are present in the dataset
#             is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

#             # if missing values are there, replace them appropriately.
#             if (is_null_present):
#                 data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation
#             #encode categorical data
#             data = preprocessor.encode_categorical_columns(data)

#             # create separate features and labels
#             X,Y=preprocessor.separate_label_feature(data,label_column_name='fraud_reported')


#             """ Applying the clustering approach"""

#             kmeans=clustering.KMeansClustering(self.file_object,self.log_writer) # object initialization.
#             number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

#             # Divide the data into clusters
#             X=kmeans.create_clusters(X,number_of_clusters)

#             #create a new column in the dataset consisting of the corresponding cluster assignments.
#             X['Labels']=Y

#             # getting the unique clusters from our dataset
#             list_of_clusters=X['Cluster'].unique()

#             """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

#             for i in list_of_clusters:
#                 cluster_data=X[X['Cluster']==i] # filter the data for one cluster

#                 # Prepare the feature and Label columns
#                 cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
#                 cluster_label= cluster_data['Labels']

#                 # splitting the data into training and test set for each cluster one by one
#                 x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
#                 # Proceeding with more data pre-processing steps
#                 x_train = preprocessor.scale_numerical_columns(x_train)
#                 x_test = preprocessor.scale_numerical_columns(x_test)


#                 model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

#                 #getting the best model for each of the clusters
#                 best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

#                 #saving the best model to the directory.
#                 file_op = file_methods.File_Operation(self.file_object,self.log_writer)
#                 save_model=file_op.save_model(best_model,best_model_name+str(i))

#             # logging the successful Training
#             self.log_writer.log(self.file_object, 'Successful End of Training')
#             self.file_object.close()

#         except Exception as e:
#             # logging the unsuccessful Training
#             self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
#             self.file_object.close()
#             raise Exception


"""
This is the Entry point for Training the Machine Learning Model.

Written By: iNeuron Intelligence
Version: 1.0
Revisions: None

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
import os


# Creating the common Logging object


# class trainModel:

#     def __init__(self, log_file=None):
#         self.log_writer = logger.App_Logger()
#         if log_file:
#             self.file_object = log_file
#         else:
#             self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

#     def trainingModel(self):
#         # Logging the start of Training
#         self.log_writer.log(self.file_object, 'Start of Training')
#         try:
#             # Getting the data from the source
#             data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
#             data=data_getter.get_data()


#             """doing the data preprocessing"""

#             preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
#             data=preprocessor.remove_columns(data,['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','age','total_claim_amount']) # remove the column as it doesn't contribute to prediction.
#             data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation

#             # check if missing values are present in the dataset
#             is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

#             # if missing values are there, replace them appropriately.
#             if (is_null_present):
#                 try:
#                     data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation
#                 except Exception:
#                     self.log_writer.log(self.file_object, 'Error in categorical imputation. Using simple imputation strategy.')
#                     # Fallback imputation strategy
#                     for col in cols_with_missing_values:
#                         if data[col].dtype.name in ['object', 'category', 'string']:
#                             # For categorical columns, fill with "Unknown"
#                             data[col] = data[col].fillna("Unknown")
#                         else:
#                             # For numerical columns, fill with mean
#                             data[col] = data[col].fillna(data[col].mean())

#             # Final check for any remaining NaN values
#             if data.isnull().sum().sum() > 0:
#                 self.log_writer.log(self.file_object, 'Found remaining NaN values after imputation. Applying final cleanup.')
#                 # For numeric columns
#                 numeric_cols = data.select_dtypes(include=['number']).columns
#                 for col in numeric_cols:
#                     if data[col].isnull().sum() > 0:
#                         data[col] = data[col].fillna(data[col].mean())

#                 # For categorical columns
#                 categorical_cols = data.select_dtypes(exclude=['number']).columns
#                 for col in categorical_cols:
#                     if data[col].isnull().sum() > 0:
#                         data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "Unknown")

#             #encode categorical data
#             data = preprocessor.encode_categorical_columns(data)

#             # create separate features and labels
#             X,Y=preprocessor.separate_label_feature(data,label_column_name='fraud_reported')


#             """ Applying the clustering approach"""

#             kmeans=clustering.KMeansClustering(self.file_object,self.log_writer) # object initialization.
#             try:
#                 # Final check for NaN values before clustering
#                 if X.isnull().sum().sum() > 0:
#                     self.log_writer.log(self.file_object, 'Found NaN values before clustering. Performing final cleanup.')
#                     X = X.fillna(X.mean())

#                 number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters
#             except Exception as e:
#                 self.log_writer.log(self.file_object, f'Error in elbow plot: {str(e)}. Using default number of clusters.')
#                 number_of_clusters = 2  # Default to 2 clusters if elbow method fails

#             # Divide the data into clusters
#             X=kmeans.create_clusters(X,number_of_clusters)

#             #create a new column in the dataset consisting of the corresponding cluster assignments.
#             X['Labels']=Y

#             # getting the unique clusters from our dataset
#             list_of_clusters=X['Cluster'].unique()

#             """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

#             for i in list_of_clusters:
#                 cluster_data=X[X['Cluster']==i] # filter the data for one cluster

#                 # Prepare the feature and Label columns
#                 cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
#                 cluster_label= cluster_data['Labels']

#                 # splitting the data into training and test set for each cluster one by one
#                 x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
#                 # Proceeding with more data pre-processing steps
#                 x_train = preprocessor.scale_numerical_columns(x_train)
#                 x_test = preprocessor.scale_numerical_columns(x_test)


#                 model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

#                 #getting the best model for each of the clusters
#                 best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

#                 #saving the best model to the directory.
#                 file_op = file_methods.File_Operation(self.file_object,self.log_writer)
#                 save_model=file_op.save_model(best_model,best_model_name+str(i))

#             # logging the successful Training
#             self.log_writer.log(self.file_object, 'Successful End of Training')
#             if hasattr(self, 'file_opened') and self.file_opened:
#                 self.file_object.close()

#         except Exception as e:
#             # logging the unsuccessful Training
#             self.log_writer.log(self.file_object, f'Unsuccessful End of Training: {str(e)}')
#             if hasattr(self, 'file_opened') and self.file_opened:
#                 self.file_object.close()
#             raise Exception

class trainModel:
    def __init__(self, log_file=None):
        self.log_writer = logger.App_Logger()
        self.file_opened = False  # Track if file is open
        if log_file:
            self.file_object = log_file
        else:
            self.file_object = "Training_Logs/ModelTrainingLog.txt"
            self.file_opened = True  # Mark file as open

    def trainingModel(self):
        try:
            # Logging the start of Training
            if self.file_opened:
                self.log_writer.log(self.file_object, 'Start of Training')

            # Getting the data from the source
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()

            """Preprocessing the Data"""
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            data = preprocessor.remove_columns(data, [
                'policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
                'incident_date', 'incident_state', 'incident_city', 'insured_hobbies', 'auto_make',
                'auto_model', 'auto_year', 'age', 'total_claim_amount'
            ])
            data.replace('?', np.NaN, inplace=True)  # Replacing '?' with NaN

            # Check for missing values
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)

            # Encoding categorical columns
            data = preprocessor.encode_categorical_columns(data)

            # Separating features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='fraud_reported')

            """ Clustering """
            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer)
            try:
                number_of_clusters = kmeans.elbow_plot(X)
            except Exception as e:
                self.log_writer.log(self.file_object, f'Error in elbow plot: {str(e)}. Defaulting to 2 clusters.')
                number_of_clusters = 2

            X = kmeans.create_clusters(X, number_of_clusters)
            X['Labels'] = Y
            list_of_clusters = X['Cluster'].unique()

            """ Finding the Best Model for Each Cluster """
            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                    random_state=355)
                x_train = preprocessor.scale_numerical_columns(x_train)
                x_test = preprocessor.scale_numerical_columns(x_test)

                model_finder = tuner.Model_Finder(self.file_object, self.log_writer)
                best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)

                file_op = file_methods.File_Operation(self.file_object, self.log_writer)
                file_op.save_model(best_model, best_model_name + str(i))

            # Logging success
            if self.file_opened:
                self.log_writer.log(self.file_object, 'Successful End of Training')
                self.file_object.close()
                self.file_opened = False  # Update flag

        except Exception as e:
            # Logging failure
            if self.file_opened:
                self.log_writer.log(self.file_object, f'Unsuccessful End of Training: {str(e)}')
                self.file_object.close()
                self.file_opened = False  # Update flag
            raise Exception
