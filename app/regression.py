import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ===============================
# Utils
# ===============================
def load_mapping(path, feature):
    df = pd.read_csv(path)
    expected_cols = {feature, f"{feature}_id"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{path} must contain {expected_cols}")
    return dict(zip(df[feature], df[f"{feature}_id"]))

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "csv"
MODEL_DIR = BASE_DIR.parent / "models/csv1"

# ===============================
# Load model
# ===============================
st.title("🚗 Predicción de precios de coches")

st.sidebar.header("Modelo ML")
model_choice = st.sidebar.selectbox(
    "Selecciona el modelo",
    ["HistGradientBoostingRegressor"]
)

model_paths = {
    "HistGradientBoostingRegressor": MODEL_DIR / "hgb_model.pkl"
}
ml_model = joblib.load(model_paths[model_choice])

# ===============================
# Load CSV y mappings
# ===============================
df = pd.read_csv(DATA_DIR / "clean_cars.csv")

brand_map = load_mapping(DATA_DIR / "mapeo_brand.csv", "brand")
model_map = load_mapping(DATA_DIR / "mapeo_model.csv", "model")
int_col_map = load_mapping(DATA_DIR / "mapeo_int_col.csv", "int_col")
ext_col_map = load_mapping(DATA_DIR / "mapeo_ext_col.csv", "ext_col")
trans_map = load_mapping(DATA_DIR / "mapeo_transmission.csv", "transmission")

# ===============================
# Columnas exactas del modelo
# ===============================
# IDs
id_cols = ["model_id","transmission_id","ext_col_id","int_col_id"]
# Numéricas
num_cols = ["model_year","milage","engine_liters","cylinders","horsepower"]
# Binarias
bin_cols = ["turbo","accident","clean_title"]
# One-hot marcas (ajusta según tu dataset)
brand_cols = [
    "brand_Alfa","brand_Aston","brand_Audi","brand_BMW","brand_Bentley","brand_Bugatti",
    "brand_Buick","brand_Cadillac","brand_Chevrolet","brand_Chrysler","brand_Dodge",
    "brand_FIAT","brand_Ferrari","brand_Ford","brand_GMC","brand_Genesis",
    "brand_Honda","brand_Hummer","brand_Hyundai","brand_INFINITI","brand_Jaguar",
    "brand_Jeep","brand_Karma","brand_Kia","brand_Lamborghini","brand_Land",
    "brand_Lexus","brand_Lincoln","brand_Lotus","brand_Lucid","brand_MINI",
    "brand_Maserati","brand_Maybach","brand_Mazda","brand_McLaren","brand_Mercedes-Benz",
    "brand_Mercury","brand_Mitsubishi","brand_Nissan","brand_Plymouth","brand_Polestar",
    "brand_Pontiac","brand_Porsche","brand_RAM","brand_Rivian","brand_Rolls-Royce",
    "brand_Saab","brand_Saturn","brand_Scion","brand_Subaru","brand_Suzuki",
    "brand_Tesla","brand_Toyota","brand_Volkswagen","brand_Volvo","brand_smart"
]
# One-hot fuel
fuel_cols = ["fuel_gasoline","fuel_diesel","fuel_electric","fuel_hybrid"]

FEATURES = id_cols + num_cols + bin_cols + brand_cols + fuel_cols

# ===============================
# Sidebar inputs
# ===============================
st.sidebar.header("Características del coche")

brand = st.sidebar.selectbox("Marca", sorted(brand_map.keys()))
brand_id = brand_map[brand]

models_available = df[df["brand"] == brand]["model"].dropna().unique()
model_name = st.sidebar.selectbox("Modelo", sorted(models_available))
model_id = model_map[model_name]

transmission = st.sidebar.selectbox("Transmisión", sorted(trans_map.keys()))
transmission_id = trans_map[transmission]

ext_col = st.sidebar.selectbox("Color exterior", sorted(ext_col_map.keys()))
ext_col_id = ext_col_map[ext_col]

int_col = st.sidebar.selectbox("Color interior", sorted(int_col_map.keys()))
int_col_id = int_col_map[int_col]

fuel = st.sidebar.radio("Tipo de combustible", ["gasoline", "diesel", "electric", "hybrid"])

model_year = st.sidebar.slider("Año del modelo", int(df.model_year.min()), int(df.model_year.max()), int(df.model_year.median()))
milage = st.sidebar.slider("Kilometraje", int(df.milage.min()), int(df.milage.max()), step=1000)
engine_liters = st.sidebar.slider("Cilindrada (L)", float(df.engine_liters.min()), float(df.engine_liters.max()), float(df.engine_liters.median()))
cylinders = st.sidebar.slider("Cilindros", int(df.cylinders.min()), int(df.cylinders.max()))
horsepower = st.sidebar.slider("Potencia (HP)", int(df.horsepower.min()), int(df.horsepower.max()))

turbo = st.sidebar.checkbox("Turbo")
accident = st.sidebar.checkbox("Historial de accidente")
clean_title = st.sidebar.checkbox("Título limpio")

# ===============================
# Build input dataframe
# ===============================
input_df = pd.DataFrame(np.zeros((1, len(FEATURES))), columns=FEATURES)

# Asignar IDs
input_df.loc[0, "model_id"] = model_id
input_df.loc[0, "transmission_id"] = transmission_id
input_df.loc[0, "ext_col_id"] = ext_col_id
input_df.loc[0, "int_col_id"] = int_col_id

# Numéricas
for col, val in {
    "model_year": model_year,
    "milage": milage,
    "engine_liters": engine_liters,
    "cylinders": cylinders,
    "horsepower": horsepower
}.items():
    input_df.loc[0, col] = val

# Binarias
for col, val in {
    "turbo": int(turbo),
    "accident": int(accident),
    "clean_title": int(clean_title)
}.items():
    input_df.loc[0, col] = val

# One-hot brand
brand_col = f"brand_{brand}"
if brand_col in input_df.columns:
    input_df.loc[0, brand_col] = 1

# One-hot fuel
fuel_col = f"fuel_{fuel}"
if fuel_col in input_df.columns:
    input_df.loc[0, fuel_col] = 1

# ===============================
# Prediction
# ===============================
st.subheader("Predicción")

if st.button("Predecir precio"):
    try:
        prediction = ml_model.predict(input_df)[0]
        st.metric("Precio estimado", f"${prediction:,.0f}")
    except ValueError as e:
        st.error(f"Error en la predicción: {e}")

    with st.expander("Ver input al modelo"):
        st.dataframe(input_df)