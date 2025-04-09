import pandas as pd
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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
            print(f"Warning: Unseen label '{val}' in {column.name}. Assigning default value {default_value}.")
            transformed.append(default_value)
    return transformed

Tk().withdraw()
file_path = askopenfilename(title="Select patient data file", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected.")
    exit()

try:
    data = pd.read_csv(file_path)
    print(f"\nFile loaded: {file_path}")
except Exception as e:
    print("Error loading file:", e)
    exit()

model, le_cognitive, le_gene_mutation_type, le_sex, le_family_history, le_motor_symptoms = joblib.load("huntington_model.pkl")

try:
    data = data[features].copy()
except KeyError as e:
    print("Missing required feature(s):", e)
    exit()

try:
    data['Cognitive_Decline'] = transform_with_unseen_handling(data['Cognitive_Decline'], le_cognitive)
    data['Gene_Mutation_Type'] = transform_with_unseen_handling(data['Gene_Mutation_Type'], le_gene_mutation_type)
    data['Sex'] = transform_with_unseen_handling(data['Sex'], le_sex)
    data['Family_History'] = transform_with_unseen_handling(data['Family_History'], le_family_history)
    data['Motor_Symptoms'] = transform_with_unseen_handling(data['Motor_Symptoms'], le_motor_symptoms)
except Exception as e:
    print("Error transforming categorical data:", e)
    exit()

data = data.fillna(data.mean(numeric_only=True))

try:
    predicted_risk = model.predict(data)[0]
    predicted_stage = get_stage_from_risk(predicted_risk)
except Exception as e:
    print("Error during prediction:", e)
    exit()

try:
    full_df = pd.read_csv("Huntington_Disease_Dataset.csv")
    stage_to_risk = {"Pre-Symptomatic": 25, "Early": 40, "Middle": 60, "Late": 85}
    full_df["Risk_Percentage"] = full_df["Disease_Stage"].map(stage_to_risk)
    population_avg = round(full_df["Risk_Percentage"].mean(), 2)
except Exception as e:
    print("Error calculating population average risk:", e)
    exit()

print("\n========================")
print(f"Predicted Stage: {predicted_stage}")
print(f"Risk Percentage: {round(predicted_risk, 2)}%")
print(f"Population Average Risk: {population_avg}%")
print("========================")
