import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("Huntington_Disease_Dataset.csv")
print("Dataset loaded successfully!\n")

print("Columns in dataset:")
print(df.columns)

stage_to_risk = {
    "Pre-Symptomatic": 25,
    "Early": 40,
    "Middle": 60,
    "Late": 85
}
df["Risk_Percentage"] = df["Disease_Stage"].map(stage_to_risk)

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
target = 'Risk_Percentage'

X = df[features].copy()
y = df[target].copy()

le_cognitive = LabelEncoder()
X['Cognitive_Decline'] = le_cognitive.fit_transform(X['Cognitive_Decline'].astype(str))

le_gene_mutation_type = LabelEncoder()
X['Gene_Mutation_Type'] = le_gene_mutation_type.fit_transform(X['Gene_Mutation_Type'].astype(str))

le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'].astype(str))

le_family_history = LabelEncoder()
X['Family_History'] = le_family_history.fit_transform(X['Family_History'].astype(str))

le_motor_symptoms = LabelEncoder()
X['Motor_Symptoms'] = le_motor_symptoms.fit_transform(X['Motor_Symptoms'].astype(str))

X = X.fillna(X.mean(numeric_only=True))
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("\nRegression model trained successfully!")

joblib.dump(
    (model, le_cognitive, le_gene_mutation_type, le_sex, le_family_history, le_motor_symptoms),
    "huntington_model.pkl"
)
print("Model and label encoders saved as 'huntington_model.pkl'")
