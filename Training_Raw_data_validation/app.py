from flask import Flask, request, jsonify
from data_validation.rawValidation import Raw_Data_validation


app = Flask(__name__)


@app.route('/validate', methods=['POST'])
def validate_data():
    try:
        # Initialize the validation class with the directory containing training files
        validator = Raw_Data_validation("Training_Batch_Files/")

        # Extract validation rules from the schema
        regex = validator.manualRegexCreation()
        LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns = validator.valuesFromSchema()

        # Perform validation
        validator.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
        validator.validateColumnLength(NumberofColumns)
        validator.validateMissingValuesInWholeColumn()

        return jsonify({"success": True, "message": "Validation successful!"})

    except Exception as e:

