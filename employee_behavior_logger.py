# employee_behavior_logger.py

import os
import csv
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "employee_logs.csv")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def log_employee_action(employee_id, action_type, claim_id=None, score=None):
    """
    Log an employee's action for internal fraud monitoring.

    Args:
        employee_id (str or int): The ID of the employee.
        action_type (str): The type of action (e.g., "register", "prediction_made", "flagged").
        claim_id (str): Optional claim or event identifier.
        score (float): Optional suspicion score.
    """
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(employee_id),
            action_type,
            claim_id or "-",
            score if score is not None else "-"
        ])


def get_employee_logs(employee_id):
    """
    Retrieve all log entries for a specific employee.

    Args:
        employee_id (str or int): The ID of the employee.

    Returns:
        list: A list of log entries for the employee.
    """
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, mode='r') as f:
        return [row for row in csv.reader(f) if row[1] == str(employee_id)]


def calculate_suspicion_score(employee_id):
    """
    Calculate a suspicion score based on the employee's historical actions.

    Returns:
        int: Suspicion score (higher = more suspicious)
    """
    logs = get_employee_logs(employee_id)
    score = 0

    # Rule 1: Excessive actions
    if len(logs) > 10:
        score += 4

    # Rule 2: Multiple flagged events
    flagged = [log for log in logs if log[2] == "flagged"]
    if len(flagged) >= 2:
        score += 3

    # Rule 3: Repeated claim IDs
    claim_ids = [log[3] for log in logs if log[3] != "-"]
    if len(claim_ids) != len(set(claim_ids)):  # duplicates exist
        score += 2

    return score
