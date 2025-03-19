# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from kneed import KneeLocator
# from file_operations import file_methods
# import os

# class KMeansClustering:
#     """
#             This class shall  be used to divide the data into clusters before training.


#             """

#     def __init__(self, file_object, logger_object):
#         self.file_object = file_object
#         self.logger_object = logger_object

#     def elbow_plot(self,data):
#         """
#                         Method Name: elbow_plot
#                         Description: This method saves the plot to decide the optimum number of clusters to the file.
#                         Output: A picture saved to the directory
#                         On Failure: Raise Exception


#                 """
#         self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')
#         wcss=[] # initializing an empty list
#         try:
#             for i in range (1,11):
#                 kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
#                 kmeans.fit(data) # fitting the data to the KMeans Algorithm
#                 wcss.append(kmeans.inertia_)
#             plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
#             plt.title('The Elbow Method')
#             plt.xlabel('Number of clusters')
#             plt.ylabel('WCSS')
#             #plt.show()
#             plt.savefig('preprocessing_data/K-Means_Elbow.PNG') # saving the elbow plot locally
#             # finding the value of the optimum cluster programmatically
#             self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
#             self.logger_object.log(self.file_object, 'The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
#             return self.kn.knee

#         except Exception as e:
#             self.logger_object.log(self.file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
#             self.logger_object.log(self.file_object,'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
#             raise Exception()

#     def create_clusters(self,data,number_of_clusters):
#         """
#                                 Method Name: create_clusters
#                                 Description: Create a new dataframe consisting of the cluster information.
#                                 Output: A datframe with cluster column
#                                 On Failure: Raise Exception

#                                 Written By: iNeuron Intelligence
#                                 Version: 1.0
#                                 Revisions: None

#                         """
#         self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
#         self.data=data
#         try:
#             self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
#             #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
#             self.y_kmeans=self.kmeans.fit_predict(data) #  divide data into clusters

#             self.file_op = file_methods.File_Operation(self.file_object,self.logger_object)
#             self.save_model = self.file_op.save_model(self.kmeans, 'KMeans') # saving the KMeans model to directory
#                                                                                     # passing 'Model' as the functions need three parameters

#             self.data['Cluster']=self.y_kmeans  # create a new column in dataset for storing the cluster information
#             self.logger_object.log(self.file_object, 'succesfully created '+str(self.kn.knee)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
#             return self.data
#         except Exception as e:
#             self.logger_object.log(self.file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
#             self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
#             raise Exception()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods
import os
import numpy as np
import pandas as pd

class KMeansClustering:
    """
    This class shall be used to divide the data into clusters before training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def handle_missing_values(self, data):
        """
        Method Name: handle_missing_values
        Description: This method handles missing values in the dataset before clustering
        Output: DataFrame with no missing values
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the handle_missing_values method of the KMeansClustering class')
        try:
            # Check if there are any missing values
            if data.isnull().sum().sum() > 0:
                # Method 1: Drop rows with missing values
                # cleaned_data = data.dropna()
                
                # Method 2: Fill missing values with the mean for each column
                cleaned_data = data.copy()
                for column in cleaned_data.columns:
                    if cleaned_data[column].dtype != 'object':  # Only fill numeric columns
                        cleaned_data[column].fillna(cleaned_data[column].mean(), inplace=True)
                
                self.logger_object.log(self.file_object, 'Successfully handled missing values in the dataset')
                return cleaned_data
            else:
                self.logger_object.log(self.file_object, 'No missing values found in the dataset')
                return data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occurred in handle_missing_values method of the KMeansClustering class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object, 'Handling missing values failed. Exited the handle_missing_values method of the KMeansClustering class')
            raise Exception()

    def elbow_plot(self, data):
        """
        Method Name: elbow_plot
        Description: This method saves the plot to decide the optimum number of clusters to the file.
        Output: A picture saved to the directory
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')
        wcss = []  # initializing an empty list
        try:
            # First, handle any missing values in the data
            cleaned_data = self.handle_missing_values(data)
            
            # Now run the elbow plot with clean data
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
                kmeans.fit(cleaned_data)  # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, 11), wcss)  # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            
            # Create directory if it doesn't exist
            os.makedirs('preprocessing_data', exist_ok=True)
            
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG')  # saving the elbow plot locally
            
            # finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(self.file_object, 'The optimum number of clusters is: ' + str(self.kn.knee) + ' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occurred in elbow_plot method of the KMeansClustering class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object, 'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        """
        Method Name: create_clusters
        Description: Create a new dataframe consisting of the cluster information.
        Output: A dataframe with cluster column
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        self.data = data
        try:
            # First, handle any missing values in the data
            cleaned_data = self.handle_missing_values(data)
            
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            self.y_kmeans = self.kmeans.fit_predict(cleaned_data)  # divide data into clusters

            self.file_op = file_methods.File_Operation(self.file_object, self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans')  # saving the KMeans model to directory

            self.data['Cluster'] = self.y_kmeans  # create a new column in dataset for storing the cluster information
            self.logger_object.log(self.file_object, 'Successfully created ' + str(number_of_clusters) + ' clusters. Exited the create_clusters method of the KMeansClustering class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occurred in create_clusters method of the KMeansClustering class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object, 'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()