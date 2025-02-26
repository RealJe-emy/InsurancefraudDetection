from flask import Flask, request, jsonify
import os
from rawValidation import Raw_Data_validation

app = Flask(__name__)

UPLOAD_FOLDER = "Training_Batch_Files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/validate', methods=['POST'])
def validate_file():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"message": "No file provided"}), 400

    file = request.files['file']

    # Check if the file has a filename
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    # Save the file to the UPLOAD_FOLDER
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    print(f"File saved to: {file_path}")

    # Initialize the Raw_Data_validation class
    validator = Raw_Data_validation(UPLOAD_FOLDER)

    # Perform validation steps
    try:
        print("Validating file name...")
        regex = validator.manualRegexCreation()
        LengthOfDateStampInFile, LengthOfTimeStampInFile, _, NumberofColumns = validator.valuesFromSchema()
        validator.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

        print("Validating column length...")
        validator.validateColumnLength(NumberofColumns)

        print("Validating missing values...")
        validator.validateMissingValuesInWholeColumn()

        print("Validation Successful!")
        return jsonify({"message": "Validation Successful!"}), 200
    except Exception as e:
        print(f"Validation Failed: {str(e)}")
        return jsonify({"message": f"Validation Failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)