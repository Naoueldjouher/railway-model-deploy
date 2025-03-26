import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Database Setup

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

########################################
# Load Model and Column Information

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

########################################
# Web Server

app = Flask(__name__)

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

    response = {"proba": proba}

    # Check if observation ID already exists
    if Prediction.select().where(Prediction.observation_id == _id).exists():
        return jsonify({
            "error": f"Observation ID {_id} already exists",
            "proba": proba
        }), 409

    # Save to database
    p = Prediction(
        observation_id=_id,
        observation=json.dumps(observation),
        proba=proba
    )

    try:
        p.save()
    except IntegrityError:
        error_msg = f"Observation ID {_id} already exists in the database."
        response["error"] = error_msg
        DB.rollback()
        return jsonify(response), 409

    return jsonify(response), 200

@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)