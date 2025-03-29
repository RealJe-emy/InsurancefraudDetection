from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
import os
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from application_logging.logger import App_Logger
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import traceback
import json
import pandas as pd
import sqlite3
from datetime import datetime
import shutil
import csv
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
dashboard.bind(app)
CORS(app)  # Enable CORS for all routes

# Define the folder to save uploaded files
UPLOAD_FOLDER = "Training_Batch_Files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOAD_FOLDER_CSV = "uploads"
os.makedirs(UPLOAD_FOLDER_CSV, exist_ok=True)

# Initialize the logger
logger = App_Logger()


# Serve the HTML files
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict.html", methods=['GET'])
@cross_origin()
def predict_page():
    return render_template('predict.html')


@app.route("/validation.html", methods=['GET'])
@cross_origin()
def validation_page():
    return render_template('validation.html')


@app.route("/train.html", methods=['GET'])
@cross_origin()
def train_page():
    return render_template('train.html')


# To validate the file
@app.route('/validate', methods=['POST'])
@cross_origin()
def validate_file():
    log_file = "Training_Logs/validationLog.txt"
    try:
        logger.log(log_file, "Request received at /validate")

        # Check if a file is included in the request
        if 'file' not in request.files:
            logger.log(log_file, "No file provided in request")
            # log_file.close()
            return jsonify({
                "status": "error",
                "message": "No file provided. Please upload a file."
            }), 400

        file = request.files['file']
        logger.log(log_file, f"File received: {file.filename}")

        # Check if the file has a filename
        if file.filename == '':
            logger.log(log_file, "Empty filename")
            # log_file.close()
            return jsonify({
                "status": "error",
                "message": "Please select a file before validating."
            }), 400

        # Save the file to the UPLOAD_FOLDER
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        logger.log(log_file, f"File saved to: {file_path}")

        # Initialize the Raw_Data_validation class
        validator = Raw_Data_validation(UPLOAD_FOLDER, log_file)

        # Stage 1: Validate file name and media type
        logger.log(log_file, "Validating file name...")
        regex = validator.manualRegexCreation()
        LengthOfDateStampInFile, LengthOfTimeStampInFile, _, _ = validator.valuesFromSchema()

        # Capture the result of file name validation
        try:
            validator.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
        except Exception as e:
            logger.log(log_file, f"File name validation failed: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Invalid file name format. The file name should follow the pattern: {regex}"
            }), 400

        # Check if the file was moved to Bad_Raw during Stage 1 validation
        bad_raw_path = os.path.join("Training_Raw_files_validated/Bad_Raw/", file.filename)
        if os.path.exists(bad_raw_path):
            logger.log(log_file, f"File failed name validation: {file.filename}")
            # log_file.close()
            return jsonify({
                "status": "error",
                "message": f"The file name format is incorrect. Please ensure it follows the required naming convention: {regex}"
            }), 400

        # Stage 2: Validate column length
        logger.log(log_file, "Validating column length...")
        _, _, _, NumberofColumns = validator.valuesFromSchema()
        try:
            validator.validateColumnLength(NumberofColumns)
        except Exception as e:
            logger.log(log_file, f"Column validation failed: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"The file contains an incorrect number of columns. Expected {NumberofColumns} columns."
            }), 400

        # Stage 3: Validate missing values
        logger.log(log_file, "Validating for missing values...")
        try:
            validator.validateMissingValuesInWholeColumn()
        except Exception as e:
            logger.log(log_file, f"Missing value validation failed: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "The file contains columns with all missing values. Please fix and try again."
            }), 400

        # Final check if the file was moved to Bad_Raw during any validation
        if os.path.exists(bad_raw_path):
            logger.log(log_file, f"File failed validation: {file.filename}")
            # log_file.close()
            return jsonify({
                "status": "error",
                "message": "The file contains invalid data. Please check for correct column count and missing values."
            }), 400

        # If the file passes all validations
        logger.log(log_file, "Validation Successful!")
        # log_file.close()
        return jsonify({
            "status": "success",
            "message": "File validation successful! Your file has been accepted for processing."
        }), 200

    except Exception as e:
        error_message = str(e)
        logger.log(log_file, f"Validation Failed: {error_message}")
        # log_file.close()

        # Provide user-friendly error messages based on the exception
        if "schema" in error_message.lower():
            message = "Schema validation error. Please check if the file format is correct."
        elif "column" in error_message.lower():
            message = "Column validation error. Please ensure the file has the correct number of columns."
        elif "missing" in error_message.lower():
            message = "Missing values detected. Please ensure all required data is present."
        elif "name" in error_message.lower():
            message = "File name error. Please ensure the file follows the naming convention."
        else:
            message = f"An unexpected error occurred during validation. Please try again."

        return jsonify({
            "status": "error",
            "message": message
        }), 500


# @app.route("/train", methods=['POST'])
# @cross_origin()
# def trainRouteClient():
#     log_file = open("Training_Logs/trainingLog.txt", "a+")
#     try:
#         logger.log(log_file, "Training request received.")
#         if request.json and 'folderPath' in request.json:
            
#             path = os.path.abspath(request.json['folderPath'])
#             print(f"Absolute path received: {path}")
#             logger.log(log_file, f"Absolute path received: {path}")
#             path = request.json['folderPath']
#             print(f"Folder path received: {path}")
#             logger.log(log_file, f"Folder path received: {path}")
            
#             # Make sure path is a directory, not a file
#             if os.path.isfile(path):
#                 path = os.path.dirname(path)  # Get the directory of the file
#                 logger.log(log_file, f"Path was a file. Using directory: {path}")
            
#             # Initialize validation with log file
#             train_valObj = train_validation(path, log_file)
#             train_valObj.train_validation()
            
#             # Initialize and train model
#             trainModelObj = trainModel(log_file)
#             trainModelObj.trainingModel()
            
#             logger.log(log_file, "Training successful.")
#             log_file.close()
#             return Response("Training successful!!")
#         else:
#             logger.log(log_file, "Error: folderPath not provided in request.")
#             log_file.close()
#             return Response("Error: folderPath is required!", status=400)
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         logger.log(log_file, f"Unexpected error: {str(e)}")
#         logger.log(log_file, f"Traceback:\n{error_trace}")
#         print(f"Unexpected error: {e}\nTraceback:\n{error_trace}")  # Print full error to terminal
#         log_file.close()
#         return Response(f"Error Occurred! {str(e)}", status=500)
    


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    log_file = "Training_Logs/trainingLog.txt"
    try:
        logger.log(log_file, "Training request received.")

        # Debug: Log request headers and form data
        logger.log(log_file, f"Request headers: {request.headers}")
        logger.log(log_file, f"Request form data: {request.form}")
        logger.log(log_file, f"Request files: {request.files}")

        # Check if the file is present in the request
        if 'file' not in request.files:
            logger.log(log_file, "Error: No file part in request.")
            return Response("Error: No file provided!", status=400)
        
        file = request.files['file']
        if file.filename == '':
            logger.log(log_file, "Error: No selected file.")
            return Response("Error: No file selected!", status=400)
            
        # Ensure the upload folder exists
        if not os.path.exists(UPLOAD_FOLDER_CSV):
            os.makedirs(UPLOAD_FOLDER_CSV)
        
        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER_CSV, file.filename)
        file.save(file_path)
        logger.log(log_file, f"File saved at: {file_path}")

        # Perform training validation
        train_valObj = train_validation(UPLOAD_FOLDER_CSV, log_file)
        train_valObj.train_validation()
        
        # Train the model
        trainModelObj = trainModel(log_file)
        trainModelObj.trainingModel()

        logger.log(log_file, "Training successful.")
        return Response("Training successful!!")
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.log(log_file, f"Unexpected error: {str(e)}")
        logger.log(log_file, f"Traceback:\n{error_trace}")
        return Response(f"Error Occurred! {str(e)}", status=500)


    # finally:
    #     if not log_file.closed:
    #         log_file.close()  # Ensure file is closed only at the end


# @app.route("/train", methods=['POST'])
# @cross_origin()
# def trainRouteClient():
#     log_file = open("Training_Logs/trainingLog.txt", "a+")
#     try:
#         logger.log(log_file, "Training request received.")
#         # Ensure a file is included in the request
#         if 'file' not in request.files:
#             logger.log(log_file, "Error: No file part in request.")
#             log_file.close()
#             return Response("Error: No file provided!", status=400)
        
#         file = request.files['file']
#         # Ensure a filename exists
#         if file.filename == '':
#             logger.log(log_file, "Error: No selected file.")
#             log_file.close()
#             return Response("Error: No file selected!", status=400)
            
#         if not os.path.exists(UPLOAD_FOLDER_CSV):
#             os.makedirs(UPLOAD_FOLDER_CSV)
#         logger.log(log_file, f"Created upload directory: {UPLOAD_FOLDER_CSV} and put {file.filename} there.")
        
#         # Define file save path
#         file_path = os.path.join(UPLOAD_FOLDER_CSV, file.filename)
#         # Save the file
#         file.save(file_path)
#         logger.log(log_file, f"File saved at: {file_path}")
        
#         # Initialize validation with the stored file directory
#         train_valObj = train_validation(UPLOAD_FOLDER_CSV, log_file)
#         train_valObj.train_validation()
        
#         # Initialize and train the model
#         # Pass the log_file to trainModel so it uses the same file
#         trainModelObj = trainModel(log_file)
#         trainModelObj.trainingModel()
        
#         logger.log(log_file, "Training successful.")
#         log_file.close()
#         return Response("Training successful!!")
        
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         try:
#             # Check if the file is closed and reopen if necessary
#             if log_file.closed:
#                 log_file = open("Training_Logs/trainingLog.txt", "a+")
#             logger.log(log_file, f"Unexpected error: {str(e)}")
#             logger.log(log_file, f"Traceback:\n{error_trace}")
#             print(f"Unexpected error: {e}\nTraceback:\n{error_trace}")  # Print full error to terminal
#         finally:
#             # Always close the file, but only if it's open
#             if not log_file.closed:
#                 log_file.close()
#         return Response(f"Error Occurred! {str(e)}", status=500)



# Predict route
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    log_file = "Prediction_Logs/PredictionLog.txt"
    try:
        logger.log(log_file, "Prediction request received")

        # Initialize variables
        pred_folder = "Prediction_Batch_Files"
        os.makedirs(pred_folder, exist_ok=True)

        # Handle different request types
        if request.files and 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logger.log(log_file, "No file selected")
                return jsonify({"status": "error", "message": "No file selected"}), 400

            filename = secure_filename(file.filename)
            file_path = os.path.join(pred_folder, filename)
            file.save(file_path)
            logger.log(log_file, f"File saved to: {file_path}")

        elif request.json and 'filepath' in request.json:
            file_path = request.json['filepath']

        elif request.form and 'filepath' in request.form:
            file_path = request.form['filepath']

        else:
            logger.log(log_file, "No valid input data provided")
            return jsonify(
                {"status": "error", "message": "Please provide either a file upload or valid file path"}), 400

        # Get the expected feature order from schema
        schema_path = "schema_prediction.json"  # Update with your actual schema path
        with open(schema_path) as f:
            schema = json.load(f)
        expected_columns = list(schema['ColName'].keys())

        # Validate and reorder columns if needed
        raw_data_path = os.path.join(pred_folder, "InputFile.csv")
        if os.path.exists(raw_data_path):
            with open(raw_data_path, 'r') as f:
                reader = csv.reader(f)
                actual_columns = next(reader)

            if set(actual_columns) != set(expected_columns):
                logger.log(log_file, "Columns don't match schema. Reordering...")
                df = pd.read_csv(raw_data_path)
                df = df[expected_columns]  # Reorder columns
                df.to_csv(raw_data_path, index=False)

        # Initialize prediction validation
        pred_val = pred_validation(pred_folder)
        pred_val.prediction_validation()

        # Initialize prediction
        pred = prediction(pred_folder)
        output_path = pred.predictionFromModel()

        # Read and return results
        results = []
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                results.append({
                    "policy_number": row[0],
                    "prediction": row[1],
                    "probability": 0.95 if row[1] == 'Y' else 0.15
                })

        return jsonify({
            "status": "success",
            "results": results,
            "summary": {
                "total_records": len(results),
                "fraud_count": sum(1 for r in results if r['prediction'] == 'Y')
            }
        })

    except Exception as e:
        logger.log(log_file, f"Prediction failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500


# single prediction endpoint
@app.route('/single_predict', methods=['POST'])
@cross_origin()
def single_predict():
    log_file = "Prediction_Logs/SinglePredictionLog.txt"
    try:
        logger.log(log_file, "Single prediction request received")

        # Get and validate input
        if not request.is_json:
            logger.log(log_file, "Request must be JSON")
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400

        data = request.get_json()
        logger.log(log_file, f"Received data: {data}")

        # Required fields (only those actually used in prediction)
        required_fields = [
            'months_as_customer', 'policy_deductable', 'policy_annual_premium',
            'incident_severity', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
            'bodily_injuries', 'property_damage', 'police_report_available'
        ]

        # Validate input
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.log(log_file, error_msg)
            return jsonify({"status": "error", "message": error_msg}), 400

        # Create input DataFrame with correct column order
        input_data = {
            'months_as_customer': int(data['months_as_customer']),
            'policy_deductable': int(data['policy_deductable']),
            'policy_annual_premium': float(data['policy_annual_premium']),
            'incident_severity': data['incident_severity'],
            'incident_hour_of_the_day': int(data['incident_hour_of_the_day']),
            'number_of_vehicles_involved': int(data['number_of_vehicles_involved']),
            'bodily_injuries': int(data['bodily_injuries']),
            'property_damage': data['property_damage'],
            'police_report_available': data['police_report_available'],
            # Include policy_number for reference (will be dropped in prediction)
            'policy_number': data.get('policy_number', 'N/A')
        }

        # Create directory if not exists
        pred_folder = "Prediction_Batch_Files"
        os.makedirs(pred_folder, exist_ok=True)

        # Save to CSV
        input_path = os.path.join(pred_folder, "InputFile.csv")
        pd.DataFrame([input_data]).to_csv(input_path, index=False)
        logger.log(log_file, f"Input data saved to {input_path}")

        # Validate and predict
        try:
            pred_val = pred_validation(pred_folder)
            pred_val.prediction_validation()
            logger.log(log_file, "Data validation completed")

            pred = prediction(pred_folder)
            output_path = pred.predictionFromModel()
            logger.log(log_file, f"Prediction completed, results at {output_path}")

            # Read results
            results = pd.read_csv(output_path)
            prediction_result = results.iloc[0]['Predictions']

            return jsonify({
                "status": "success",
                "prediction": prediction_result,
                "probability": 0.95 if prediction_result == 'Y' else 0.15,
                "policy_number": input_data['policy_number'],
                "important_factors": get_important_factors(data)
            })

        except Exception as e:
            logger.log(log_file, f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"status": "error", "message": f"Prediction processing failed: {str(e)}"}), 500

    except Exception as e:
        logger.log(log_file, f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


#helper function
def get_important_factors(data):
    """Identify important factors contributing to the prediction"""
    factors = []

    # Add your business logic for important factors
    if float(data.get('total_claim_amount', 0)) > 10000:
        factors.append("High claim amount")
    if int(data.get('number_of_vehicles_involved', 1)) > 1:
        factors.append("Multiple vehicles involved")
    if data.get('police_report_available', 'NO') == 'NO':
        factors.append("No police report")
    if data.get('property_damage', 'NO') == 'YES':
        factors.append("Property damage reported")

    return factors if factors else ["No significant risk factors identified"]


# @app.route("/train", methods=['POST'])
# @cross_origin()
# def trainRouteClient():
#     try:
#         if request.json['folderPath'] is not None:
#             path = request.json['folderPath']
#             train_valObj = train_validation(path)  # object initialization
#             train_valObj.train_validation()  # calling the training_validation function

#             trainModelObj = trainModel()  # object initialization
#             trainModelObj.trainingModel()  # training the model for the files in the table

#     except ValueError:
#         return Response("Error Occurred! %s" % ValueError)
#     except KeyError:
#         return Response("Error Occurred! %s" % KeyError)
#     except Exception as e:
#         return Response("Error Occurred! %s" % e)
#     return Response("Training successful!!")

# @app.route("/train", methods=['POST'])
# @cross_origin()
# def trainRouteClient():
#     log_file = open("Training_Logs/trainingLog.txt", "a+")
#     try:
#         logger.log(log_file, "Training request received.")
        
#         if 'folderPath' not in request.json:
#             logger.log(log_file, "No folder path provided in request.")
#             log_file.close()
#             return jsonify({"status": "error", "message": "No folder path provided."}), 400
        
#         path = request.json['folderPath']
#         logger.log(log_file, f"Folder path received: {path}")
        
#         # Validate training data
#         train_valObj = train_validation(path)
#         logger.log(log_file, "Starting training data validation...")
#         train_valObj.train_validation()
#         logger.log(log_file, "Training data validation completed successfully.")
        
#         # Train model
#         trainModelObj = trainModel()
#         logger.log(log_file, "Starting model training...")
#         trainModelObj.trainingModel()
#         logger.log(log_file, "Model training completed successfully.")
        
#         log_file.close()
#         return jsonify({"status": "success", "message": "Training successful!"}), 200
    
#     except ValueError as e:
#         logger.log(log_file, f"ValueError occurred: {str(e)}")
#         log_file.close()
#         return jsonify({"status": "error", "message": str(e)}), 400
#     except KeyError as e:
#         logger.log(log_file, f"KeyError occurred: {str(e)}")
#         log_file.close()
#         return jsonify({"status": "error", "message": str(e)}), 400
#     except Exception as e:
#         logger.log(log_file, f"Unexpected error: {str(e)}")
#         log_file.close()
#         return jsonify({"status": "error", "message": "An unexpected error occurred during training."}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(port=port, debug=True)