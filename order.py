import pandas as pd
# from application_logging.logger import App_Logger

# # Initialize the logger
# logger = App_Logger()

def reorder_prediction_columns(data):


    """
    Reorders the columns in the prediction data to match the order used during training.
    
    Parameters:
    -----------
    prediction_data : pandas.DataFrame
        The data to be used for prediction
    
    Returns:
    --------
    pandas.DataFrame
        The prediction data with columns reordered to match training data
    """
    # Hardcoded training column order
    training_columns_order = [
        'policy_csl', 'insured_sex', 'insured_education_level',
        'incident_severity', 'property_damage', 'police_report_available',
        'insured_occupation_armed-forces', 'insured_occupation_craft-repair',
        'insured_occupation_exec-managerial', 'insured_occupation_farming-fishing',
        'insured_occupation_handlers-cleaners', 'insured_occupation_machine-op-inspct',
        'insured_occupation_other-service', 'insured_occupation_priv-house-serv',
        'insured_occupation_prof-specialty', 'insured_occupation_protective-serv', 
        'insured_occupation_sales', 'insured_occupation_tech-support',
        'insured_occupation_transport-moving', 'insured_relationship_not-in-family',
        'insured_relationship_other-relative', 'insured_relationship_own-child',
        'insured_relationship_unmarried', 'insured_relationship_wife',
        'incident_type_Parked Car', 'incident_type_Single Vehicle Collision',
        'incident_type_Vehicle Theft', 'collision_type_Rear Collision',
        'collision_type_Side Collision', 'authorities_contacted_Fire',
        'authorities_contacted_Other', 'authorities_contacted_Police',
        'months_as_customer', 'policy_deductable', 'policy_annual_premium',
        'umbrella_limit', 'capital-gains', 'capital-loss',
        'incident_hour_of_the_day', 'number_of_vehicles_involved',
        'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
        'vehicle_claim'
    ]
    
    # Check if all required columns exist in the prediction data
    missing_columns = [col for col in training_columns_order if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Prediction data is missing these columns: {missing_columns}")
    
    # Check for extra columns in prediction data
    extra_columns = [col for col in data.columns if col not in training_columns_order]
    if extra_columns:
        print(f"Warning: Prediction data contains extra columns that will be ignored: {extra_columns}")
    
    # Reorder columns to match training order
    data = data[training_columns_order].copy()
    
    return data


def add_policy_numbers_to_csv(policy_numbers, file_path):
            try:
                # Read the existing CSV file
                df = pd.read_csv(file_path)

                # Ensure the number of policy numbers matches the number of rows
                print(f"Length of policy_numbers: {len(policy_numbers)} and length of df: {len(df)}")
                # if len(policy_numbers) != len(df):
                #     raise ValueError("The number of policy numbers does not match the number of rows in the CSV file.")

                # Add the policy_numbers as a new column
                df['policy_number'] = policy_numbers

                # Save the updated DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                print(f"Policy numbers added to file: {file_path}")
            except Exception as e:
                print(f"Error adding policy numbers to CSV: {str(e)}")
                raise
