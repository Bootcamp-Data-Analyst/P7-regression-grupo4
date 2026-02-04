import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ===============================
# Config
# ===============================
st.set_page_config(page_title="Predicción coches", layout="wide")
st.title("🚗 Predicción de precios de coches")

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "csv"
MODEL_DIR = BASE_DIR.parent / "models/csv1"

# ===============================
# Load model
# ===============================
model_path = MODEL_DIR / "hgb_model.pkl"
ml_model = joblib.load(model_path)

# Columnas EXACTAS usadas en training
FEATURES = list(ml_model.feature_names_in_)

# ===============================
# Load data
# ===============================
df = pd.read_csv(DATA_DIR / "clean_cars.csv")

def load_mapping(path, col):
    m = pd.read_csv(path)
    return dict(zip(m[col], m[f"{col}_id"]))

brand_map = load_mapping(DATA_DIR / "mapeo_brand.csv", "brand")
model_map = load_mapping(DATA_DIR / "mapeo_model.csv", "model")
ext_col_map = load_mapping(DATA_DIR / "mapeo_ext_col.csv", "ext_col")
int_col_map = load_mapping(DATA_DIR / "mapeo_int_col.csv", "int_col")
trans_map = load_mapping(DATA_DIR / "mapeo_transmission.csv", "transmission")

# ===============================
# Sidebar inputs
# ===============================
st.sidebar.header("Características")

brand = st.sidebar.selectbox("Marca", sorted(brand_map))
models_available = df[df["brand"] == brand]["model"].dropna().unique()
model_name = st.sidebar.selectbox("Modelo", sorted(models_available))

transmission = st.sidebar.selectbox("Transmisión", sorted(trans_map))
ext_col = st.sidebar.selectbox("Color exterior", sorted(ext_col_map))
int_col = st.sidebar.selectbox("Color interior", sorted(int_col_map))

fuel = st.sidebar.radio(
    "Combustible",
    ["gasoline", "diesel", "electric", "hybrid"]
)

model_year = st.sidebar.slider(
    "Año",
    int(df.model_year.min()),
    int(df.model_year.max()),
    int(df.model_year.median())
)

milage = st.sidebar.slider(
    "Kilometraje",
    0,
    int(df.milage.max()),
    50000,
    step=1000
)

engine_liters = st.sidebar.slider(
    "Cilindrada (L)",
    float(df.engine_liters.min() + 1),
    float(df.engine_liters.max()),
    float(df.engine_liters.median())
)

cylinders = st.sidebar.slider(
    "Cilindros",
    int(df.cylinders.min() + 1),
    int(df.cylinders.max())
)

horsepower = st.sidebar.slider(
    "Potencia (HP)",
    int(df.horsepower.min() + 1),
    int(df.horsepower.max())
)

turbo = st.sidebar.checkbox("Turbo")
accident = st.sidebar.checkbox("Accidente")
clean_title = st.sidebar.checkbox("Título limpio")

# ===============================
# Build input EXACTO
# ===============================
input_df = pd.DataFrame(0, index=[0], columns=FEATURES)

# Numéricas
for col, val in {
    "model_year": model_year,
    "milage": milage,
    "engine_liters": engine_liters,
    "cylinders": cylinders,
    "horsepower": horsepower,
}.items():
    if col in input_df:
        input_df.loc[0, col] = val

# Binarias
for col, val in {
    "turbo": int(turbo),
    "accident": int(accident),
    "clean_title": int(clean_title),
}.items():
    if col in input_df:
        input_df.loc[0, col] = val

# IDs (si existen en el modelo)
id_features = {
    "model_id": model_map.get(model_name),
    "transmission_id": trans_map.get(transmission),
    "ext_col_id": ext_col_map.get(ext_col),
    "int_col_id": int_col_map.get(int_col),
}

for col, val in id_features.items():
    if col in input_df:
        input_df.loc[0, col] = val

# One-hot brand
brand_col = f"brand_{brand}"
if brand_col in input_df:
    input_df.loc[0, brand_col] = 1

# One-hot fuel
fuel_col = f"fuel_{fuel}"
if fuel_col in input_df:
    input_df.loc[0, fuel_col] = 1

# ===============================
# Prediction
# ===============================
st.subheader("Predicción")

if st.button("Predecir precio"):
    raw_pred = ml_model.predict(input_df)[0]

    # 🔥 CLAVE: el modelo está en log
    price = np.exp(raw_pred)

    st.metric("Precio estimado", f"${price:,.0f}")

    with st.expander("Input enviado al modelo"):
        st.dataframe(input_df.T)