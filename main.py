from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, send_from_directory, g
from flask_cors import CORS, cross_origin
from functools import wraps
import os
import traceback
import json
import pandas as pd
import sqlite3
import shutil
import csv
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets

from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from application_logging.logger import App_Logger
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction
import flask_monitoringdashboard as dashboard
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

# Add this after app initialization
REQUIRED_TEMPLATES = ['index.html', 'login.html', 'registration.html',
                     'train.html', 'predict.html', 'validate.html',
                     '404.html', '500.html']

@app.before_first_request
def check_templates():
    missing = []
    for template in REQUIRED_TEMPLATES:
        if not os.path.exists(os.path.join(app.template_folder, template)):
            missing.append(template)
    if missing:
        raise RuntimeError(f"Missing required templates: {', '.join(missing)}")


def get_db():
    """Get a thread-local database connection"""
    if 'db' not in g:
        g.db = sqlite3.connect('database.db')
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
        g.db.execute("PRAGMA journal_mode = WAL")
    return g.db


def init_db():
    """Initialize database tables"""
    with app.app_context():
        db = get_db()
        try:
            db.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    role TEXT DEFAULT 'employee'
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS employee_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER NOT NULL,
                    session_token TEXT NOT NULL,
                    expires_at DATETIME NOT NULL,
                    FOREIGN KEY (employee_id) REFERENCES employees(id)
                )
            """)
            db.commit()
        except Exception as e:
            print(f"Database initialization error: {e}")
            db.rollback()


@app.teardown_appcontext
def close_db(exception=None):
    """Close the database at the end of each request"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize the logger
logger = App_Logger()

@app.errorhandler(sqlite3.Error)
def handle_db_errors(e):
    return jsonify({"error": "Database operation failed"}), 500


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # For API requests, check Authorization header
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        # For browser requests, also check cookies
        if not token and request.cookies:
            token = request.cookies.get('authToken')

        if not token:
            # If it's an API request (expecting JSON)
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({"error": "Authorization required"}), 401
            # If it's a browser request
            else:
                return redirect(url_for('login_page', next=request.path))

        db = get_db()
        user = db.execute(
            "SELECT e.id, e.username, e.email, e.role FROM employee_sessions s "
            "JOIN employees e ON s.employee_id = e.id "
            "WHERE s.session_token = ? AND s.expires_at > datetime('now')",
            (token,)
        ).fetchone()

        if not user:
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({"error": "Invalid token"}), 401
            else:
                return redirect(url_for('login_page', next=request.path))

        request.user = dict(user)
        return f(*args, **kwargs)

    return decorated_function


@app.route('/debug-routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'route': str(rule),
            'methods': sorted(rule.methods),
            'endpoint': rule.endpoint
        })
    return jsonify(routes)

@app.route("/", methods=['GET'])
@app.route("/index.html", methods=['GET'])
@cross_origin()
def home():
    # Check for auth token in headers or cookies
    token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.cookies.get('authToken')

    if token:
        try:
            db = get_db()
            user = db.execute(
                "SELECT e.id, e.username, e.email FROM employee_sessions s "
                "JOIN employees e ON s.employee_id = e.id "
                "WHERE s.session_token = ? AND s.expires_at > datetime('now')",
                (token,)
            ).fetchone()

            if user:
                # Don't redirect, just show index.html to authenticated users too
                return render_template('index.html')
        except Exception as e:
            app.logger.error(f"Token verification error: {str(e)}")

    return render_template('index.html')  # Show regular index for unauthenticated users

# Serve the HTML files
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/registration.html')
def register_page():
    return render_template('registration.html')
@app.route('/train.html')
@login_required
def train_page():
    return render_template('train.html')
@app.route('/predict.html')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/validation.html')
@login_required
def validation_page():
    return render_template('validation.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/protected')
@login_required
def protected():
    db = get_db()  # Safe connection
    user_data = db.execute(
        "SELECT * FROM employees WHERE email = ?",
        (request.employee['email'],)
    ).fetchone()

    return jsonify(dict(user_data))


# Auth Service Class
class AuthService:
    def register_user(self, username, email, password):
        db = get_db()  # Get fresh connection per call
        try:
            # Check if email exists
            existing = db.execute(
                "SELECT id FROM employees WHERE email = ?",
                (email,)
            ).fetchone()

            if existing:
                return None, "Email already registered"

            # Hash password and create user
            hashed_pw = generate_password_hash(password)
            db.execute(
                "INSERT INTO employees (username, email, password_hash) "
                "VALUES (?, ?, ?)",
                (username, email, hashed_pw)
            )
            db.commit()
            return email, None
        except Exception as e:
            db.rollback()
            return None, str(e)


# Auth API Endpoints
@app.route('/api/auth/register', methods=['POST'])
@cross_origin()
def register():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ['username', 'email', 'password']

    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

    db = get_db()  # 🔴 ADD THIS LINE - Get fresh connection
    try:
        # Check if email exists
        existing = db.execute(
            "SELECT id FROM employees WHERE email = ?",
            (data['email'],)
        ).fetchone()

        if existing:
            return jsonify({"error": "Email already registered"}), 400

        # Hash password and create user
        hashed_pw = generate_password_hash(data['password'])
        db.execute(
            "INSERT INTO employees (username, email, password_hash) "
            "VALUES (?, ?, ?)",
            (data['username'], data['email'], hashed_pw)
        )
        db.commit()
        return jsonify({"message": "Registration successful"}), 201
        from employee_behavior_logger import log_employee_action, calculate_suspicion_score

        # Log initial registration activity
        user_id = db.execute("SELECT id FROM employees WHERE email = ?", (data['email'],)).fetchone()['id']
        claim_id = f"REG_{user_id}"

        log_employee_action(user_id, "register", claim_id)

        # Optional: calculate initial behavior score
        score = calculate_suspicion_score(user_id)
        if score >= 7:
            log_employee_action(user_id, "flagged", claim_id, score)


    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
@cross_origin()
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.cookies.get('authToken')
    if not token:
        return jsonify({"message": "Already logged out"}), 200

    db = get_db()
    db.execute("DELETE FROM employee_sessions WHERE session_token = ?", (token,))
    db.commit()

    response = jsonify({"message": "Logged out successfully"})
    response.delete_cookie('authToken')
    return response

@app.route('/api/auth/login', methods=['POST'])
@cross_origin()
def login():
    data = request.get_json()
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Email and password required"}), 400

    db = get_db()
    try:
        # Get user with all necessary fields
        user = db.execute(
            "SELECT id, username, email, role, password_hash FROM employees WHERE email = ?",
            (data['email'],)
        ).fetchone()

        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        user_dict = dict(user)

        if not check_password_hash(user_dict['password_hash'], data['password']):
            return jsonify({"error": "Invalid email or password"}), 401

        # Create session token
        token = secrets.token_hex(32)
        expires_at = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

        # Delete any existing sessions for this user
        db.execute(
            "DELETE FROM employee_sessions WHERE employee_id = ?",
            (user_dict['id'],)
        )

        # Create new session
        db.execute(
            "INSERT INTO employee_sessions (employee_id, session_token, expires_at) "
            "VALUES (?, ?, ?)",
            (user_dict['id'], token, expires_at)
        )
        db.commit()

        response = jsonify({
            "token": token,
            "user": {
                "id": user_dict['id'],
                "username": user_dict['username'],
                "email": user_dict['email'],
                "role": user_dict['role']
            },
            "redirect": "/index.html"
        })
        response.set_cookie(
            "authToken",
            token,
            httponly=True,  # Prevents JS access
            samesite='Lax',  # Basic protection
            max_age=60 * 60 * 24 * 7  # 7 days
        )
        return response

    except Exception as e:
        db.rollback()
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "An error occurred during login"}), 500


@app.route('/api/auth/verify', methods=['GET'])
@cross_origin()
def verify_token():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({"valid": False}), 401

    db = get_db()
    try:
        session = db.execute(
            "SELECT e.id, e.username, e.email, e.role "
            "FROM employee_sessions s "
            "JOIN employees e ON s.employee_id = e.id "
            "WHERE s.session_token = ? AND s.expires_at > datetime('now')",
            (token,)
        ).fetchone()

        return jsonify({
            "valid": bool(session),
            "user": dict(session) if session else None
        }), 200

    except Exception as e:
        app.logger.error(f"Token verification error: {str(e)}")
        return jsonify({"valid": False}), 500

# Protected route example
@app.route('/api/auth/me', methods=['GET'])
@cross_origin()
def get_current_user():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db()  # 🔴 ADD THIS LINE
    user = db.execute(
        "SELECT e.id, e.username, e.email FROM employee_sessions s "
        "JOIN employees e ON s.employee_id = e.id "
        "WHERE s.session_token = ? AND s.expires_at > datetime('now')",
        (token,)
    ).fetchone()

    if not user:
        return jsonify({"error": "Invalid token"}), 401

    return jsonify(dict(user))


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
@app.route('/api/predict', methods=['POST'])
@cross_origin()
@login_required
def predictRouteClient():
    log_file = "Prediction_Logs/PredictionLog.txt"
    try:
        logger.log(log_file, "Prediction request received")

        pred_folder = "Prediction_Batch_Files"
        os.makedirs(pred_folder, exist_ok=True)

        # Handle file upload or filepath input
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
            return jsonify({
                "status": "error",
                "message": "Please provide either a file upload or valid file path"
            }), 400

        # =============================
        # Actual prediction logic here
        # =============================

        # Validate and insert prediction data
        pred_val = pred_validation(raw_data=file_path)
        pred_val.prediction_validation()

        predictor = prediction()
        prediction_output_path = predictor.predict_from_model()

        logger.log(log_file, f"Prediction successful. Output saved to: {prediction_output_path}")

        return jsonify({
            "status": "success",
            "message": "Prediction completed",
            "output_file": prediction_output_path
        }), 200

    except Exception as e:
        logger.log(log_file, f"Prediction failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

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