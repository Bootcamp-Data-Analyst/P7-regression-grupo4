import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime as dt

# Carpeta donde está este script (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models") 

# --- Título ---
st.title("Predicción de precios de coches")

# --- Selección de modelo ---
st.sidebar.header("Selecciona el modelo")
model_choice = st.sidebar.selectbox(
    "Modelo",
    ("RandomForest", "GradientBoosting", "HistGradientBoostingRegressor")
)

# --- Cargar modelo según selección ---
model_paths = {
    "RandomForest": os.path.join(MODEL_DIR, "rf_model.pkl"),
    "GradientBoosting": os.path.join(MODEL_DIR, "gbr_model.pkl"),
    "HistGradientBoostingRegressor": os.path.join(MODEL_DIR, "hgb_model.pkl")
}
model = joblib.load(model_paths[model_choice])

# --- Entrada de usuario ---
st.sidebar.header("Características del coche")

brands = [
    'Alfa', 'Aston', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Buick', 'Cadillac', 
    'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari', 'Ford', 'GMC', 'Genesis', 
    'Honda', 'Hummer', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Karma', 'Kia', 
    'Lamborghini', 'Land', 'Lexus', 'Lincoln', 'Lotus', 'Lucid', 'MINI', 'Maserati', 
    'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz', 'Mercury', 'Mitsubishi', 'Nissan', 
    'Plymouth', 'Polestar', 'Pontiac', 'Porsche', 'RAM', 'Rivian', 'Rolls-Royce', 
    'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 
    'Volvo', 'smart'
]

# Para multi-selección
selected_brands = st.multiselect("Selecciona una o varias marcas", brands)

# Inicializar todas las columnas de marca a 0
brand_columns = {f'brand_{b}': 0 for b in brands}

# Kilometraje como rango
milage_range = st.sidebar.slider(
    "Kilometraje",
    min_value=0,
    max_value=500_000,
    value=(50_000, 150_000),
    step=1000
)
kms = np.arange(milage_range[0], milage_range[1]+1, 5000)

# Motor, cilindros, accidentes, turbo, caballos
engine_liters = st.sidebar.selectbox(
    "Tamaño del motor (L)",
    np.round(np.arange(0.65, 8.5, 0.1), 2)
)
cylinders = st.sidebar.selectbox("Cilindros", [3,4,6,8])
accident = st.sidebar.selectbox("Accidente", [0,1])
clean_title = st.sidebar.selectbox("Título limpio", [0,1])
turbo = st.sidebar.selectbox("Turbo", [0,1])
horsepower = st.sidebar.number_input("Caballos", 0, 800)

# Año de matriculación
model_date = st.sidebar.date_input(
    "Fecha de matriculación",
    min_value=dt.date(1974, 1, 1),
    max_value=dt.date(2025, 12, 31),
    value=dt.date(2015, 1, 1)
)
model_year = int(model_date.year)

# --- Combustible ---
fuel_diesel = int(st.sidebar.checkbox("Diésel"))
fuel_gasoline = int(st.sidebar.checkbox("Gasolina"))
fuel_hybrid = int(st.sidebar.checkbox("Híbrido"))
fuel_electric = int(st.sidebar.checkbox("Eléctrico"))

# --- Crear DataFrame de entrada para todas las filas de kilometraje ---
dfs = []
for km in kms:
    row = {
        'model_year': model_year,
        'milage': km,
        'accident': accident,
        'clean_title': clean_title,
        'engine_liters': engine_liters,
        'cylinders': cylinders,
        'turbo': turbo,
        'horsepower': horsepower,
        'fuel_diesel': fuel_diesel,
        'fuel_gasoline': fuel_gasoline,
        'fuel_hybrid': fuel_hybrid,
        'fuel_electric': fuel_electric,
        **brand_columns
    }
    dfs.append(row)

input_data = pd.DataFrame(dfs)


# Reordenar columnas según modelo
input_data = input_data[model.feature_names_in_]

# --- Predicción ---
pred_log = model.predict(input_data)
pred_price = np.expm1(pred_log)  # revertir log1p si se aplicó

st.subheader("Precio predicho:")
st.write(f"€ {pred_price[0]:,.0f}")

# --- Gráfico Predicción vs Real ---
if st.checkbox("Mostrar gráfico Predicción vs Real"):
    try:
        y_test_real = joblib.load(os.path.join(MODEL_DIR, "y_test_real.pkl"))
        pred_paths = {
            "RandomForest": os.path.join(MODEL_DIR, "y_pred_rf.pkl"),
            "GradientBoosting": os.path.join(MODEL_DIR, "y_pred_gbr.pkl"),
            "HistGradientBoostingRegressor": os.path.join(MODEL_DIR, "y_pred_hgb.pkl")
        }
        y_pred_model = joblib.load(pred_paths[model_choice])

        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_test_real, y_pred_model, alpha=0.3)
        ax.plot([0, 200_000], [0, 200_000], linestyle='--', color='red')
        ax.set_xlabel("Precio real (€)")
        ax.set_ylabel("Precio predicho (€)")
        ax.set_title(f"Predicción vs Real ({model_choice})")
        st.pyplot(fig)
    except:
        st.write("Necesitas definir y_test_real y y_pred_gbr en tu sesión para ver el gráfico.")
