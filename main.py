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

# Serve the HTML file at the root URL
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

# Validate uploaded file
@app.route('/validate', methods=['POST'])
@cross_origin()
def validate_file():
    try:
        # Open a log file to track the validation process
        log_file = open("Training_Logs/validationLog.txt", "a+")
        logger.log(log_file, "Request received at /validate")

        # Check if a file is included in the request
        if 'file' not in request.files:
            logger.log(log_file, "No file provided in request")
            log_file.close()
            return jsonify({"message": "No file provided"}), 400

        file = request.files['file']
        logger.log(log_file, f"File received: {file.filename}")

        # Check if the file has a filename
        if file.filename == '':
            logger.log(log_file, "No selected file")
            log_file.close()
            return jsonify({"message": "No selected file"}), 400

        # Save the file to the UPLOAD_FOLDER
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        logger.log(log_file, f"File saved to: {file_path}")

        # Initialize the Raw_Data_validation class
        validator = Raw_Data_validation(UPLOAD_FOLDER, log_file)

        # Perform validation steps
        logger.log(log_file, "Validating file name...")
        regex = validator.manualRegexCreation()
        LengthOfDateStampInFile, LengthOfTimeStampInFile, _, NumberofColumns = validator.valuesFromSchema()
        validator.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

        logger.log(log_file, "Validating column length...")
        validator.validateColumnLength(NumberofColumns)

        logger.log(log_file, "Validating missing values...")
        validator.validateMissingValuesInWholeColumn()

        logger.log(log_file, "Validation Successful!")
        log_file.close()
        return jsonify({"message": "Validation Successful!"}), 200

    except Exception as e:
        logger.log(log_file, f"Validation Failed: {str(e)}")
        log_file.close()
        return jsonify({"message": f"Validation Failed: {str(e)}"}), 500


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
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            train_valObj = train_validation(path)  # object initialization
            train_valObj.train_validation()  # calling the training_validation function

            trainModelObj = trainModel()  # object initialization
            trainModelObj.trainingModel()  # training the model for the files in the table

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