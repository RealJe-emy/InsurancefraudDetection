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
from order import add_policy_numbers_to_csv

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
        print(f"Expected columns: {expected_columns}")
        logger.log(log_file, f"Expected columns: {expected_columns}")

        # Validate and reorder columns if needed
        raw_data_path = os.path.join(pred_folder, "fraudDetection_extended.csv")
        print(raw_data_path)
        try:
            if os.path.exists(raw_data_path):
                with open(raw_data_path, 'r') as f:
                    reader = csv.reader(f)
                    actual_columns = next(reader)
                    print(f"Actual columns: {actual_columns}")
                    logger.log(log_file, f"Actual columns: {actual_columns}")

                if set(actual_columns) != set(expected_columns):
                    logger.log(log_file, "Columns don't match schema. Reordering...")
                    df = pd.read_csv(raw_data_path)
                    df = df[expected_columns]  # Reorder columns
                    df.to_csv(raw_data_path, index=True)
        except Exception as e:
            logger.log(log_file, f"Error reordering columns: {str(e)}")
            return jsonify({"status": "error", "message": f"Error reordering columns: {str(e)}"}), 500
        
        missing = [col for col in expected_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in expected_columns]
        print(f"Missing columns: {missing}")
        print(f"Extra columns: {extra}")

        logger.log(log_file, "Column reordering completed")
        # Extract policy_number from the dataframe
        policy_numbers = df['policy_number'].tolist()
        print(f"length of policy_numbers is : {len(policy_numbers)}")
        logger.log(log_file, f"Extracted policy numbers: {policy_numbers}")

        # Initialize prediction validation
        pred_val = pred_validation(pred_folder)
        pred_val.prediction_validation()

        # Initialize prediction
        pred = prediction(pred_folder)
        output_path = pred.predictionFromModel()
        print(f"Tis is the output path file {output_path}")

        # Read and return results
        results = []
        add_policy_numbers_to_csv(policy_numbers, output_path)
        logger.log(log_file, f"Policy numbers added to output file: {output_path}")
        print(f"Policy numbers added to output file: {output_path}")
        
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                results.append({
                    "policy_number": row[2],
                    "prediction": row[0],
                    "probability": row[1]
                })

        return jsonify({
            "status": "success",
            "results": results,
            "summary": {
                "total_records": len(results),
                "fraud_count": sum(1 for r in results if r['prediction'] == 'Y')
            },
            "policy_numbers": policy_numbers
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
        logger.log(log_file, "\n" + "=" * 80)
        logger.log(log_file, "=== NEW PREDICTION REQUEST ===")
        logger.log(log_file, f"Timestamp: {datetime.now().isoformat()}")

        # ===== 1. REQUEST VALIDATION =====
        if not request.is_json:
            error_msg = "Request must be JSON"
            logger.log(log_file, error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg,
                "resolution": "Set Content-Type to application/json"
            }), 400

        # ===== 2. DATA PARSING =====
        try:
            data = request.get_json()
            logger.log(log_file, "Raw JSON data received")
        except Exception as e:
            error_msg = f"JSON parsing failed: {str(e)}"
            logger.log(log_file, error_msg)
            return jsonify({
                "status": "error",
                "message": "Invalid JSON format",
                "details": str(e)
            }), 400

        # ===== 3. INPUT VALIDATION =====
        required_fields = {
            'months_as_customer': int,
            'policy_deductable': int,
            'policy_annual_premium': float,
            'incident_severity': str,
            'incident_hour_of_the_day': int,
            'number_of_vehicles_involved': int,
            'bodily_injuries': int,
            'property_damage': str,
            'police_report_available': str
        }

        validation_errors = []
        input_data = {'policy_number': data.get('policy_number', 'N/A')}

        for field, field_type in required_fields.items():
            if field not in data:
                validation_errors.append(f"Missing field: {field}")
                continue
            try:
                input_data[field] = field_type(data[field])
            except (ValueError, TypeError):
                validation_errors.append(
                    f"Invalid type for {field}: expected {field_type.__name__}, got {type(data[field]).__name__}"
                )

        if validation_errors:
            logger.log(log_file, "Input validation failed:")
            logger.log(log_file, "\n".join(validation_errors))
            return jsonify({
                "status": "error",
                "message": "Input validation failed",
                "errors": validation_errors,
                "required_fields": list(required_fields.keys())
            }), 400

        # ===== 4. DATA PREPARATION =====
        pred_folder = "Prediction_Batch_Files"
        bad_data_folder = os.path.join(pred_folder, "Bad_Data")
        os.makedirs(pred_folder, exist_ok=True)
        os.makedirs(bad_data_folder, exist_ok=True)

        input_path = os.path.join(pred_folder, "InputFile.csv")
        try:
            input_df = pd.DataFrame([input_data])
            input_df.to_csv(input_path, index=False)
            logger.log(log_file, f"Input data saved to {input_path}")
        except Exception as e:
            error_msg = f"Failed to save input data: {str(e)}"
            logger.log(log_file, error_msg)
            return jsonify({
                "status": "error",
                "message": "Data preparation failed",
                "details": str(e)
            }), 500

        # ===== 5. DATA VALIDATION =====
        try:
            pred_val = pred_validation(pred_folder)

            # Add debug logging for validation
            logger.log(log_file, "Starting data validation...")
            validation_result = pred_val.prediction_validation()

            # Check if file was moved to Bad_Data
            bad_data_path = os.path.join(bad_data_folder, "InputFile.csv")
            if os.path.exists(bad_data_path):
                error_msg = "Data validation failed - file moved to Bad_Data"
                logger.log(log_file, error_msg)

                # Read validation log for details
                validation_log = os.path.join("Prediction_Logs", "Prediction_Validation_Log.txt")
                validation_details = ""
                if os.path.exists(validation_log):
                    with open(validation_log, 'r') as f:
                        validation_details = f.read().splitlines()[-5:]  # Get last 5 lines

                return jsonify({
                    "status": "error",
                    "message": "Data validation failed",
                    "details": validation_details,
                    "resolution": "Check your input data against schema requirements"
                }), 400

            logger.log(log_file, "Data validation passed")
        except Exception as e:
            error_msg = f"Validation process failed: {str(e)}"
            logger.log(log_file, error_msg)
            return jsonify({
                "status": "error",
                "message": "Data validation system error",
                "details": str(e)
            }), 500

        # ===== 6. PREDICTION EXECUTION =====
        try:
            logger.log(log_file, "Starting prediction...")
            pred = prediction(pred_folder)

            # Add debug logging for model files
            model_dir = "models/"
            logger.log(log_file, f"Model directory contents: {os.listdir(model_dir)}")

            output_path = pred.predictionFromModel()
            logger.log(log_file, f"Prediction completed, results at: {output_path}")

            # ===== 7. RESULT PROCESSING =====
            try:
                results = pd.read_csv(output_path)
                if results.empty:
                    raise ValueError("Empty prediction results")

                prediction_result = results.iloc[0]['Predictions']
                confidence = 0.95 if prediction_result == 'Y' else 0.15
                factors = get_important_factors(input_data)

                return jsonify({
                    "status": "success",
                    "prediction": prediction_result,
                    "confidence": confidence,
                    "factors": factors,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                error_msg = f"Result processing failed: {str(e)}"
                logger.log(log_file, error_msg)
                return jsonify({
                    "status": "error",
                    "message": "Could not process results",
                    "details": str(e)
                }), 500

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.log(log_file, error_msg)

            # Check for common prediction errors
            if "KMeans" in str(e):
                return jsonify({
                    "status": "error",
                    "message": "Model loading failed",
                    "details": "KMeans model not found or invalid",
                    "resolution": "Please retrain your models"
                }), 503

            elif "Cluster" in str(e):
                return jsonify({
                    "status": "error",
                    "message": "Cluster prediction failed",
                    "details": str(e),
                    "resolution": "Check your training data distribution"
                }), 500

            else:
                return jsonify({
                    "status": "error",
                    "message": "Prediction processing failed",
                    "details": str(e) if app.debug else None,
                    "resolution": "Contact support with error details"
                }), 500

    except Exception as e:
        error_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.log(log_file, f"UNHANDLED ERROR [{error_id}]: {str(e)}\n{traceback.format_exc()}")

        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error_id": error_id,
            "resolution": "Contact support with this error ID"
        }), 500


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


# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(port=port, debug=True)