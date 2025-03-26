import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import Model, IntegerField, FloatField, TextField, IntegrityError, PostgresqlDatabase
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

########################################
# Database Setup

# Connect to PostgreSQL on Railway
DATABASE_URL = os.getenv('DATABASE_URL')
DB = PostgresqlDatabase(DATABASE_URL, sslmode='require')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


# Initialize database and create tables
def initialize_db():
    try:
        DB.connect()
        DB.create_tables([Prediction], safe=True)
    except Exception as e:
        app.logger.error(f"Error connecting to the database: {e}")
        raise e

# Close DB connection after request
@app.teardown_appcontext
def close_db(error):
    if not DB.is_closed():
        DB.close()

########################################
# Load Model and Column Information

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

########################################
# Helper function for input validation
def validate_observation(obs):
    """
    Validate the observation dictionary to ensure:
    - 'age' is an integer
    - 'education' is a string
    - 'hours-per-week' is an integer
    - 'native-country' is a string
    """
    required_fields = ["age", "education", "hours-per-week", "native-country"]

    # Check if all required fields are present
    for field in required_fields:
        if field not in obs:
            return f"Missing field: {field}"

    # Check data types
    if not isinstance(obs["age"], int):
        return "Invalid data type for 'age'. Expected integer."
    if not isinstance(obs["education"], str):
        return "Invalid data type for 'education'. Expected string."
    if not isinstance(obs["hours-per-week"], int):
        return "Invalid data type for 'hours-per-week'. Expected integer."
    if not isinstance(obs["native-country"], str):
        return "Invalid data type for 'native-country'. Expected string."

    return None  # No errors

########################################
# Web Server Routes

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()

    # Validate JSON structure
    if "id" not in obs_dict or "observation" not in obs_dict:
        return jsonify({"error": "Missing 'id' or 'observation' in request"}), 400

    _id = obs_dict["id"]
    observation = obs_dict["observation"]

    # Validate observation format
    error_msg = validate_observation(observation)
    if error_msg:
        return jsonify({"error": error_msg}), 400

    # Convert input into DataFrame
    obs_df = pd.DataFrame([observation], columns=columns).astype(dtypes)

    # Make prediction
    proba = pipeline.predict_proba(obs_df)[0, 1]

    # Check if observation ID already exists
    if Prediction.select().where(Prediction.observation_id == _id).exists():
        # If observation ID already exists, just return the 'proba' value without any error
        return jsonify({"proba": proba}), 200

    # Save to database
    p = Prediction(
        observation_id=_id,
        observation=json.dumps(observation),
        proba=proba
    )

    try:
        p.save()
    except IntegrityError:
        # This is a safeguard; ideally won't hit due to the previous check.
        response = {"proba": proba}
        DB.rollback()
        return jsonify(response), 409

    # Return only the 'proba' value in the response
    return jsonify({"proba": proba}), 200

@app.route('/health', methods=['GET'])

@app.route('/update', methods=['POST'])
def update():
    obs_dict = request.get_json()

    # Validate JSON structure
    if "id" not in obs_dict or "true_class" not in obs_dict:
        return jsonify({"error": "Missing 'id' or 'true_class' in request"}), 400

    _id = obs_dict["id"]
    true_class = obs_dict["true_class"]

    # Check if observation ID exists
    prediction = Prediction.select().where(Prediction.observation_id == _id).first()
    if not prediction:
        return jsonify({"error": f"Prediction with id {_id} not found"}), 404

    # Update the true_class
    prediction.true_class = true_class

    try:
        prediction.save()
        return jsonify({
            "observation_id": _id,
            "observation": prediction.observation,  # Ensure this exists in the DB
            "true_class": true_class,
            "proba": prediction.proba
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update true_class: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to ensure the app and database are working."""
    try:
        # Simple DB check
        DB.get_tables()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

########################################
# Main entry point

if __name__ == '__main__':
    # Initialize DB and run the app
    try:
        initialize_db()
    except Exception as e:
        app.logger.error("Failed to initialize the database.")
        exit(1)  # Exit the app if DB initialization fails

    app.run(debug=True, host='0.0.0.0', port=5000)
