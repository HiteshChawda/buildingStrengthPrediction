import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("new_dataset.csv")

target = ["Strength", "Durability"]

X = df.drop(["Strength", "Durability", "Load_Bearing_Capacity"], axis=1)
y = df[target]

label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

train_feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

st.set_page_config(page_title="Building Strength & Durability Predictor", layout="wide")

st.title("Building Strength & Durability Predictor")
st.write("Enter building details to predict *Strength*, *Durability*, and *Safety Rating*.")


Material_Quality = st.slider("Material Quality (1-10)", 1, 10, 7)

c1, c2, c3 = st.columns(3)
Concrete_Grade = c1.selectbox("Concrete Grade", ["M20", "M30", "M40"])
Steel_Grade = c2.selectbox("Steel Grade", ["Fe415", "Fe500"])
Age_of_Building = c3.number_input("Age of Building (years)", 0, 200, 20)

c4, c5 = st.columns(2)
No_of_Floors = c4.number_input("Number of Floors", 1, 50, 3)
Foundation_Depth = c5.number_input("Foundation Depth (3–20 ft)", 1.0, 30.0, 10.0)

c6, c7, c8 = st.columns(3)
Soil_Type = c6.selectbox("Soil Type", ["Sandy", "Clay", "Loam", "Rocky"])
Seismic_Zone = c7.selectbox("Seismic Zone", ["I", "II", "III", "IV", "V"])
Moisture_Exposure = c8.selectbox("Moisture Exposure", ["Low", "Medium", "High"])
Maintenance_Frequency = st.slider("Maintenance Frequency (per year)", 0, 5, 1)


user_data = {
    "Material_Quality": Material_Quality,
    "Concrete_Grade": Concrete_Grade,
    "Steel_Grade": Steel_Grade,
    "Age_of_Building": Age_of_Building,
    "No_of_Floors": No_of_Floors,
    "Foundation_Depth": Foundation_Depth,
    "Soil_Type": Soil_Type,
    "Seismic_Zone": Seismic_Zone,
    "Moisture_Exposure": Moisture_Exposure,
    "Maintenance_Frequency": Maintenance_Frequency
}

user_df = pd.DataFrame([user_data])


for col in user_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        user_df[col] = le.transform(user_df[col])

user_df = user_df[train_feature_names]


if st.button("Predict Strength & Durability"):

    prediction = model.predict(user_df)
    strength_pred = prediction[0][0]
    durability_pred = prediction[0][1]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Strength:** {strength_pred:.2f}")
    st.write(f"**Predicted Durability:** {durability_pred:.2f}")

    def classify_strength(value):
        if value > 110:
            return "Very High Strength"
        elif value > 95:
            return "High Strength"
        elif value > 80:
            return "Moderate Strength"
        else:
            return "Low Strength"

    def classify_durability(value):
        if value > 80:
            return "Very High Durability"
        elif value > 60:
            return "High Durability"
        elif value > 40:
            return "Moderate Durability"
        else:
            return "Low Durability"

    def building_safety_rating(strength, durability):
        if strength > 100 and durability > 70:
            return "A (Excellent & Very Safe)"
        elif strength > 85 and durability > 50:
            return "B (Good & Safe)"
        elif strength > 75 and durability > 40:
            return "C (Moderate Safety)"
        else:
            return "D (Low Safety Needs Attention)"

    strength_label = classify_strength(strength_pred)
    durability_label = classify_durability(durability_pred)
    overall_rating = building_safety_rating(strength_pred, durability_pred)

    st.subheader("Building Condition Report")
    st.write(f"**Strength Category:** {strength_label}")
    st.write(f"**Durability Category:** {durability_label}")
    st.write(f"**Overall Safety Rating:** {overall_rating}")

    st.subheader("Explanation")
    if overall_rating.startswith("A"):
        st.info("This building is structurally excellent with high lifespan and stability.")
    elif overall_rating.startswith("B"):
        st.info("This building is strong and safe with good long-term durability.")
    elif overall_rating.startswith("C"):
        st.warning("This building is moderately safe but may need improvements.")
    elif overall_rating.startswith("D"):
        st.error("This building may have safety concerns and needs urgent structural evaluation.")
