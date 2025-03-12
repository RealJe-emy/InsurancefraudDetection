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

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')
dashboard.bind(app)
CORS(app)  # Enable CORS for all routes

# Define the folder to save uploaded files
UPLOAD_FOLDER = "Training_Batch_Files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    log_file = open("Training_Logs/validationLog.txt", "a+")
    try:
        logger.log(log_file, "Request received at /validate")

        # Check if a file is included in the request
        if 'file' not in request.files:
            logger.log(log_file, "No file provided in request")
            log_file.close()
            return jsonify({
                "status": "error",
                "message": "No file provided. Please upload a file."
            }), 400

        file = request.files['file']
        logger.log(log_file, f"File received: {file.filename}")

        # Check if the file has a filename
        if file.filename == '':
            logger.log(log_file, "Empty filename")
            log_file.close()
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
            log_file.close()
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
            log_file.close()
            return jsonify({
                "status": "error",
                "message": "The file contains invalid data. Please check for correct column count and missing values."
            }), 400

        # If the file passes all validations
        logger.log(log_file, "Validation Successful!")
        log_file.close()
        return jsonify({
            "status": "success",
            "message": "File validation successful! Your file has been accepted for processing."
        }), 200

    except Exception as e:
        error_message = str(e)
        logger.log(log_file, f"Validation Failed: {error_message}")
        log_file.close()

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


# Predict route
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = pred_validation(path)  # object initialization
            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization
            path = pred.predictionFromModel()  # predicting for dataset present in database
            return Response("Prediction File created at %s!!!" % path)

        elif request.form is not None:
            path = request.form['filepath']

            pred_val = pred_validation(path)  # object initialization
            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization
            path = pred.predictionFromModel()  # predicting for dataset present in database
            return Response("Prediction File created at %s!!!" % path)

        elif request.files is not None and 'file' in request.files:
            file = request.files['file']

            # Create Prediction_Batch_Files directory if it doesn't exist
            pred_folder = "Prediction_Batch_Files"
            os.makedirs(pred_folder, exist_ok=True)

            # Save uploaded file
            file_path = os.path.join(pred_folder, file.filename)
            file.save(file_path)

            # Initialize prediction validation
            pred_val = pred_validation(pred_folder)
            pred_val.prediction_validation()

            # Make prediction
            pred = prediction(pred_folder)
            path = pred.predictionFromModel()

            return Response("Prediction File created at %s!!!" % path)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


# Train route
@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json and 'folderPath' in request.json:
            path = request.json['folderPath']

        elif request.form and 'folderPath' in request.form:
            path = request.form['folderPath']

        elif request.files and 'file' in request.files:
            file = request.files['file']

            # Create directory if it doesn't exist
            train_folder = "Training_Batch_Files"
            os.makedirs(train_folder, exist_ok=True)

            # Save uploaded file
            file_path = os.path.join(train_folder, file.filename)
            file.save(file_path)

            path = train_folder
        else:
            return Response("No data provided for training")

        # Perform validation
        train_valObj = train_validation(path)
        train_valObj.train_validation()

        # Train model
        trainModelObj = trainModel()
        trainModelObj.trainingModel()

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")


# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(port=port, debug=True)