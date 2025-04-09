from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = "secretkey"
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

model, le_cognitive, le_sex, le_family_history, le_motor_symptoms, le_gene_mutation_type = joblib.load("huntington_model.pkl")

features = [
    'Age',
    'HTT_CAG_Repeat_Length',
    'Cognitive_Decline',
    'Chorea_Score',
    'Brain_Volume_Loss',
    'Functional_Capacity',
    'HTT_Gene_Expression_Level',
    'Protein_Aggregation_Level',
    'Sex',
    'Family_History',
    'Motor_Symptoms',
    'Gene_Mutation_Type'
]

def get_stage_from_risk(risk):
    if risk < 30:
        return "Pre-Symptomatic"
    elif risk < 50:
        return "Early"
    elif risk < 70:
        return "Middle"
    else:
        return "Late"

def transform_with_unseen_handling(column, encoder, default_value=-1):
    """Transform a column using the encoder, replacing unseen labels with a default value."""
    transformed = []
    for val in column:
        if val in encoder.classes_:
            transformed.append(encoder.transform([val])[0])
        else:
            flash(f"Warning: Unseen label '{val}' in {column.name}. Assigned default value {default_value}.")
            transformed.append(default_value)
    return transformed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    risk_percent = None
    pop_avg_risk = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                data = pd.read_csv(file)
                data = data[features].copy()
            except KeyError as e:
                flash(f"Missing required feature columns: {e}")
                return redirect(request.url)
            except Exception as e:
                flash(f"Error reading file: {e}")
                return redirect(request.url)

            try:
                data['Cognitive_Decline'] = transform_with_unseen_handling(data['Cognitive_Decline'], le_cognitive)
                data['Sex'] = transform_with_unseen_handling(data['Sex'], le_sex)
                data['Family_History'] = transform_with_unseen_handling(data['Family_History'], le_family_history)
                data['Motor_Symptoms'] = transform_with_unseen_handling(data['Motor_Symptoms'], le_motor_symptoms)
                data['Gene_Mutation_Type'] = transform_with_unseen_handling(data['Gene_Mutation_Type'], le_gene_mutation_type)
            except Exception as e:
                flash(f"Error transforming categorical data: {e}")
                return redirect(request.url)

            data = data.fillna(data.mean(numeric_only=True))

            try:
                predicted_risk = model.predict(data)[0]
                predicted_stage = get_stage_from_risk(predicted_risk)
                risk_percent = round(predicted_risk, 2)
                prediction = predicted_stage
            except Exception as e:
                flash(f"Error during prediction: {e}")
                return redirect(request.url)

            try:
                df = pd.read_csv("Huntington_Disease_Dataset.csv")
                stage_to_risk = {
                    "Pre-Symptomatic": 25,
                    "Early": 40,
                    "Middle": 60,
                    "Late": 85
                }
                df["Risk_Percentage"] = df["Disease_Stage"].map(stage_to_risk)
                pop_avg_risk = round(df["Risk_Percentage"].mean(), 2)
            except Exception as e:
                flash(f"Error calculating population average risk: {e}")
                return redirect(request.url)

    return render_template("index.html", prediction=prediction, risk_percent=risk_percent, pop_avg_risk=pop_avg_risk)

@app.route("/consult")
def consult():
    return render_template("consult.html")

if __name__ == "__main__":
    app.run(debug=True)
