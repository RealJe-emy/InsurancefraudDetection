from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import os

class Model_Finder:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.sv_classifier = SVC()
        self.xgb = XGBClassifier(objective='binary:logistic', n_jobs=-1)

    def get_best_params_for_svm(self, train_x, train_y):
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # Handle missing values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
            train_x = imputer.fit_transform(train_x)

            # Initializing with different combinations of parameters
            self.param_grid = {
                "kernel": ['rbf', 'sigmoid'],
                "C": [0.1, 0.5, 1.0],
                "random_state": [0, 100, 200, 300]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.sv_classifier, param_grid=self.param_grid, cv=5, verbose=3)

            # Finding the best parameters
            self.grid.fit(train_x, train_y)

            # Extracting the best parameters
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.random_state = self.grid.best_params_['random_state']

            # Creating a new model with the best parameters
            self.sv_classifier = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)

            # Training the new model
            self.sv_classifier.fit(train_x, train_y)

            self.logger_object.log(self.file_object,
                                   'SVM best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_svm method of the Model_Finder class')

            return self.sv_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_params_for_svm method of the Model_Finder class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object,
                                   'SVM training failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

    # def get_best_params_for_xgboost(self, train_x, train_y):
    #     self.logger_object.log(self.file_object,
    #     'Entered the get_best_params_for_xgboost method of the Model_Finder class')
    #     try:
    #         # Handle missing values using SimpleImputer
    #         imputer = SimpleImputer(strategy='mean')
    #         train_x = imputer.fit_transform(train_x)
            
    #         # Initializing with appropriate XGBoost parameters
    #         # Note: 'criterion' is not a parameter for XGBClassifier, it's for sklearn's tree-based models
    #         self.param_grid_xgboost = {
    #             "n_estimators": [100, 130],
    #             "max_depth": range(3, 10, 2),  # XGBoost typically uses smaller depths
    #             "learning_rate": [0.01, 0.1, 0.3],
    #             "subsample": [0.8, 1.0],
    #             "colsample_bytree": [0.8, 1.0]
    #         }
            
    #         # Create XGBoost classifier
    #         self.xgb = XGBClassifier(eval_metric='logloss')
            
    #         # Creating an object of the Grid Search class
    #         self.grid = GridSearchCV(self.xgb, self.param_grid_xgboost, verbose=3, cv=5)
            
    #         # Finding the best parameters
    #         self.grid.fit(train_x, train_y)
            
    #         # Extracting the best parameters
    #         best_params = self.grid.best_params_
            
    #         # Creating a new model with the best parameters
    #         self.xgb = XGBClassifier(
    #             n_estimators=best_params['n_estimators'],
    #             max_depth=best_params['max_depth'],
    #             learning_rate=best_params['learning_rate'],
    #             subsample=best_params['subsample'],
    #             colsample_bytree=best_params['colsample_bytree'],
    #             eval_metric='logloss'
    #         )
            
    #         # Training the new model
    #         self.xgb.fit(train_x, train_y)
            
    #         self.logger_object.log(self.file_object,
    #         'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            
    #         return self.xgb
            
    #     except Exception as e:
    #         self.logger_object.log(self.file_object,
    #         'Exception occurred in get_best_params_for_xgboost method of the Model_Finder class. Exception message: ' + str(e))
    #         self.logger_object.log(self.file_object,
    #         'XGBoost Parameter tuning failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
    #         # Provide the actual exception rather than just raising a generic one
    #         raise Exception(f"XGBoost parameter tuning failed: {str(e)}")

    def get_best_params_for_xgboost(self, train_x, train_y):
        self.logger_object.log(self.file_object,
        'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # Handle missing values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            train_x_imputed = imputer.fit_transform(train_x)
            
            # Create a basic XGBoost model with default parameters first
            xgb_default = XGBClassifier(objective='binary:logistic',  
                                    eval_metric='logloss')
            
            # Train the default model
            xgb_default.fit(train_x_imputed, train_y)
            
            self.logger_object.log(self.file_object,
                                'Successfully trained XGBoost with default parameters')
            
            # Return the model without grid search for now
            # You can manually implement parameter tuning later
            return xgb_default
            
        except Exception as e:
            self.logger_object.log(self.file_object,
            'Exception occurred in get_best_params_for_xgboost method of the Model_Finder class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object,
            'XGBoost Parameter tuning failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            
            # Fall back to a Random Forest classifier if XGBoost fails
            self.logger_object.log(self.file_object, 'Falling back to Random Forest classifier')
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(train_x_imputed, train_y)
            return rf

    def get_best_model(self, train_x, train_y, test_x, test_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        try:
            # Handle missing values in test data
            imputer = SimpleImputer(strategy='mean')  # Use the same strategy as in training
            train_x = imputer.fit_transform(train_x)
            test_x = imputer.transform(test_x)

            # Create best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)

            if len(test_y.unique()) == 1:
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))

            # Create best model for SVM
            self.svm = self.get_best_params_for_svm(train_x, train_y)
            self.prediction_svm = self.svm.predict(test_x)

            if len(test_y.unique()) == 1:
                self.svm_score = accuracy_score(test_y, self.prediction_svm)
                self.logger_object.log(self.file_object, 'Accuracy for SVM:' + str(self.svm_score))
            else:
                self.svm_score = roc_auc_score(test_y, self.prediction_svm)
                self.logger_object.log(self.file_object, 'AUC for SVM:' + str(self.svm_score))

            # Comparing the two models
            if self.svm_score < self.xgboost_score:
                return 'XGBoost', self.xgboost
            else:
                return 'SVM', self.sv_classifier

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_model method of the Model_Finder class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()