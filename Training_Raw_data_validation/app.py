from flask import Flask, request, jsonify
import os
from rawValidation import Raw_Data_validation

app = Flask(__name__)

UPLOAD_FOLDER = "Training_Batch_Files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/validate', methods=['POST'])
def validate_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Initialize validation class and perform validation
    validator = Raw_Data_validation(UPLOAD_FOLDER)
    regex = validator.manualRegexCreation()
    LengthOfDateStampInFile, LengthOfTimeStampInFile, _, NumberofColumns = validator.valuesFromSchema()

    try:
        validator.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
        validator.validateColumnLength(NumberofColumns)
        validator.validateMissingValuesInWholeColumn()
        return jsonify({"message": "Validation Successful!"}), 200
    except Exception as e:
        return jsonify({"message": f"Validation Failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
