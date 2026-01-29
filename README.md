<h1 align="center">Predicción de precios de coches usados — EDA, optimización y dashboard</h1>

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/6838e7d4-2aca-4d31-b8cc-5f211cc0a53f"
    alt="Predicción de precios de coches usados - Dashboard"
    width="420"
    style="
      max-width: 90%;
      height: auto;
      border-radius: 14px;
      box-shadow: 0 10px 24px rgba(0,0,0,0.18);
      border: 1px solid rgba(0,0,0,0.08);
    "
  />
</p>

<p align="center">
  <sub><b>EDA</b> · Optimización · Dashboard interactivo</sub>
</p>



# 1. 🧠 Resumen ejecutivo

Proyecto de regresión para predecir el precio de coches usados a partir de datos reales (Cars.com), con un enfoque orientado a producto.

El trabajo cubre todo el pipeline de un proyecto real de Data Science:

- 📊 EDA y análisis de mercado  
- 🧹 Limpieza y feature engineering  
- 🤖 Modelado con SVR + validación K-Fold  
- ⚙️ Optimización automática con Optuna  
- 🌐 Demo interactiva en Streamlit  
- 📈 Dashboard analítico para mercado + modelo  

El objetivo principal es construir un **modelo robusto, explicable y productivizable**, priorizando la generalización, la trazabilidad de decisiones y la utilidad real del producto final.


---

# 2. 🎯 Objetivos técnicos

- Diseñar un pipeline reproducible de limpieza, transformación y entrenamiento.
- Minimizar overfitting mediante validación cruzada y control explícito del gap.
- Optimizar hiperparámetros sin sesgo mediante Optuna + K-Fold.
- Construir un producto usable para predicción real.
- Diseñar un dashboard que explique tanto el mercado como el comportamiento del modelo.


---

# 3. Estructura del repositorio

```
📦 proyecto-7-grupo-4-prediccion-precios-coches-svr-dashboard/
┣ 📁 data/         # dataset original y dataset limpio
┣ 📁 notebooks/    # EDA + entrenamiento + evaluación
┣ 📁 src/          # utilidades (limpieza, features, training, etc.)
┣ 📁 app/          # Streamlit (predicción + feedback)
┣ 📁 dashboard/    # Power BI (.pbix) o assets (capturas)
┣ 📄 requirements.txt
┗ 📄 README.md
```

---

# 4. Tecnologías utilizadas

- **Python:** NumPy, pandas, scikit-learn  
- **Optimización:** Optuna  
- **Visualización:** matplotlib, seaborn  
- **Entorno:** VS Code + entorno virtual  
- **Producto:** Streamlit  
- **Dashboard:** Power BI / Streamlit  ?????????
- **Control de versiones:** Git + GitHub  

---

# 5. Cómo ejecutar el proyecto (entorno virtual)

## Requisitos
- Python 3.11.7 (recomendado)
- Git instalado

---

<details>
<summary><strong>✅ Resultado final</strong></summary>

## 1. Clonar el repositorio
```bash
git clone https://github.com/Bootcamp-Data-Analyst/proyecto-7-grupo-4-prediccion-precios-coches-dashboard.git
cd proyecto-7-grupo-4-prediccion-precios-coches-svr-dashboard
```

---

## 2. Crear y activar el entorno virtual

### Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

---

## 3. Instalar dependencias
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Ejecutar la demo (Streamlit)
```bash
streamlit run app/Home.py
```

---

## 5. Ejecutar notebooks (opcional)

Abre los notebooks desde VS Code o Jupyter apuntando al intérprete del entorno `.venv`:

- Carpeta: `notebooks/`
- Kernel / intérprete: `.venv`
</details>

---

# 6. Dataset y variables (análisis descriptivo y transformación)

## 6.1 Dataset original

**Fuente:** Kaggle — Used Car Price Prediction Dataset (Cars.com)  
**Tamaño:** 4.009 registros  
**Objetivo (target):** `price`

El dataset original contiene información estructurada y semiestructurada sobre vehículos usados, combinando variables numéricas, categóricas y campos de texto.

**Variables originales:**

- `brand`
- `model`
- `model_year`
- `milage`
- `fuel_type`
- `engine`
- `transmission`
- `ext_col`
- `int_col`
- `accident`
- `clean_title`
- `price`

---

<details>
<summary><strong>📌 Tipología inicial de variables</strong></summary>

- **Numéricas:**  
  `model_year`, `milage`, `price`

- **Categóricas nominales:**  
  `brand`, `model`, `fuel_type`, `transmission`, `ext_col`, `int_col`

- **Texto semiestructurado:**  
  `engine`

- **Binarias / booleanas incompletas:**  
  `accident`, `clean_title`

</details>

---

<details>
<summary><strong>⚠️ Problemas detectados en el dataset original</strong></summary>

Durante el EDA inicial se identificaron varios problemas típicos en datasets reales:

- **Inconsistencias de formato:**
  - `milage` y `price` almacenados como texto.
  - `engine` como campo semiestructurado con múltiples patrones.

- **Alta cardinalidad:**
  - `model`, `engine`, `transmission`, `ext_col`, `int_col`.

- **Naming inconsistente:**
  - Múltiples valores distintos para conceptos equivalentes (especialmente en `transmission` y colores).

- **Nulos y valores ambiguos:**
  - `fuel_type`, `accident`, `clean_title`.

Estos factores hacían inviable el uso directo del dataset sin un proceso de limpieza y transformación.

</details>

---

## 6.2 Dataset modificado (dataset final de trabajo)

Tras el proceso de limpieza, normalización e ingeniería de variables, se obtiene un dataset más compacto, consistente y adecuado para modelado.

**Variables finales:**

- `brand`
- `model_year`
- `milage`
- `fuel_type` (reducido a 4 valores)
- `fuel_type_num`
- `transmission` (normalizada)
- `ext_col` (normalizada)
- `int_col` (normalizada)
- `accident` (booleano: 0 / 1)
- `clean_title` (booleano: 0 / 1)
- `price`
- `engine_liters`
- `cylinders`
- `turbo`

---

## 6.3 Transformaciones aplicadas

<details>
<summary><strong>🧹 Limpieza de tipos y parsing</strong></summary>

- `price`:  
  De texto `"$15,000"` → entero `15000`.

- `milage`:  
  De texto `"110,000 mi."` → entero `110000`.

- `model_year`:  
  Convertido a entero.

</details>

---

<details>
<summary><strong>🧹 Imputación y tratamiento de valores nulos</strong></summary>

Durante el proceso de limpieza se identificaron valores nulos en tres variables: `fuel_type`, `accident` y `clean_title`. El tratamiento se realizó de forma **selectiva y justificada**, priorizando coherencia semántica y calidad del dataset.

### `fuel_type` y `accident`

- Las filas con valores nulos fueron **eliminadas**.
- Motivo:
  - Representan un **porcentaje reducido** del dataset.
  - La imputación podía introducir **ruido o sesgo artificial**.
  - La eliminación no compromete la representatividad global.

### `clean_title`

- Los valores nulos **no se consideran ausencia de dato**, sino equivalentes semánticos de `False`.
- Se realizó **imputación explícita a `False`** y posterior conversión a variable binaria:
  - `0` → no clean title  
  - `1` → clean title  

- Justificación:
  - La propia lógica del dataset indica que solo se marca explícitamente el valor positivo.
  - Mantener los nulos habría introducido ambigüedad y pérdida de información.

### Resultado

Este tratamiento permite:

- Evitar imputaciones artificiales.
- Reducir ruido.
- Mantener coherencia semántica.
- Preservar la calidad estadística del dataset final.

</details>

---

<details>
<summary><strong>🎨 Normalización de variables categóricas</strong></summary>

### `fuel_type`

- Reducción a **4 categorías principales**:
  - gasoline
  - diesel
  - hybrid
  - electric

- Creación de versión numérica:
  - `fuel_type_num` → representación ordinal para facilitar análisis y modelado.

---

### `transmission`

- Normalización de valores semánticamente equivalentes:
  - Ejemplo:  
    `"8-Speed Automatic"`, `"8-Speed A/T"`, `"Automatic 8-Speed"` → `automatic_8`

Objetivo: reducir ruido semántico y cardinalidad artificial.

---

### `ext_col` y `int_col`

- Normalización y agrupación a **colores base**:
  - Ejemplos:
    - `"Jet Black"`, `"Black Metallic"`, `"Glossy Black"` → `black`
    - `"Dark Gray"`, `"Graphite"` → `gray`

Objetivo: reducir ruido, colisiones semánticas y dimensionalidad innecesaria.

</details>

---

<details>
<summary><strong>🔘 Variables booleanas</strong></summary>

### `accident`

- Conversión a variable binaria:
  - `0` → sin accidente
  - `1` → con accidente

---

### `clean_title`

- Imputación explícita de valores nulos como `False`.
- Conversión a variable binaria:
  - `0` → no clean title
  - `1` → clean title

Objetivo: evitar ambigüedad semántica y tratar correctamente el patrón de missingness.

</details>

---

<details>
<summary><strong>⚙️ Ingeniería de variables desde <code>engine</code></strong></summary>

A partir del campo semiestructurado `engine`, se extrajeron variables numéricas con mayor valor predictivo:

- `engine_liters`
- `cylinders`
- `turbo` (0 / 1)

Ejemplo:
"2.0L I4 Turbo" → engine_liters=2.0, cylinders=4, turbo=1"

Este enfoque permite capturar información mecánica relevante evitando el uso directo de texto libre.

</details>

---

<details>
<summary><strong>🧠 Eliminación de la variable <code>model</code> (criterio técnico)</strong></summary>

La variable `model` presentaba **una cardinalidad extremadamente alta**:

- **1.898 valores únicos** para un dataset de **4.009 registros**.
- Esto implica que **casi la mitad de los registros tienen un modelo único o muy poco frecuente**.

### Problemas derivados

- **Explosión dimensional** al aplicar one-hot encoding.
- **Matriz extremadamente dispersa (sparse)**.
- **Riesgo alto de overfitting**, ya que el modelo aprende patrones casi únicos por fila.
- **Baja capacidad de generalización**: el modelo no puede aprender relaciones estables cuando la mayoría de categorías apenas se repiten.
- **Coste computacional elevado** sin ganancia real de información.

### Alternativas evaluadas

- **One-hot encoding completo:** descartado por dimensionalidad extrema.
- **Agrupación por frecuencia:** viable, pero implicaba una fuerte pérdida semántica.
- **Frequency / target encoding:** introducía riesgo de leakage y complejidad adicional.

### Decisión final

Se decidió **eliminar completamente la variable `model`**, priorizando:

- Estabilidad del dataset.
- Generalización del modelo.
- Interpretabilidad.
- Simplicidad y robustez del pipeline.

Esta decisión se considera **coherente y defendible** dada la relación entre:

> Tamaño del dataset ↔ cardinalidad de la variable.

</details>

---

<details>
<summary><strong>✅ Resultado final</strong></summary>

El dataset transformado:

- Presenta **tipos homogéneos**
- Reduce **ruido semántico**
- Controla **cardinalidad excesiva**
- Mejora la **capacidad de generalización**
- Mantiene **interpretabilidad**

Este proceso permite trabajar con un conjunto de datos **más estable, coherente y realista para proyectos de regresión** en entornos productivos.

</details>

---

# 7. Planteamiento y selección del modelo

En esta fase no se fija un modelo concreto, sino que se define el **marco de decisión** para seleccionar posteriormente el algoritmo más adecuado, alineando **la naturaleza del problema**, **las características del dataset** y **los requisitos del producto final**.
 

---


## 7.1 Naturaleza del problema

El objetivo es **predecir una variable continua (`price`)**, por lo que se trata de un problema de **regresión supervisada**.

Aspectos clave:

- Relaciones **no lineales** entre variables.
- Presencia de **outliers naturales** del mercado.
- Dataset de tamaño **medio (~4k registros)**.
- Mezcla de variables numéricas, categóricas y transformadas.

---

## 7.2 Requisitos del modelo

El modelo debe:

- Capturar **no linealidad**.
- Generalizar correctamente (**control de overfitting**).
- Ser **computacionalmente eficiente**.
- Permitir **análisis e interpretación**.
- Integrarse fácilmente en la aplicación final.

---

## 7.3 Estrategia de evaluación

Los modelos se evaluarán mediante:

- **K-Fold Cross-Validation**
- Métricas: **RMSE, MAE y R²**
- Análisis de estabilidad y residuos

El objetivo es seleccionar el modelo con **mejor equilibrio entre rendimiento, estabilidad e interpretabilidad**.

---

## 7.4 Modelos candidatos

Se evaluarán distintas familias de modelos:

- **Lineales:** Linear, Ridge, Lasso  
- **Árboles:** Random Forest, Gradient Boosting  
- **Kernel:** Support Vector Regression (SVR)  
- **Boosting avanzado:** XGBoost, LightGBM, CatBoost (si procede)

---

## 7.5 Criterio de selección

La elección final se basará en:

- Rendimiento en validación cruzada
- Estabilidad entre folds
- Gap de overfitting
- Análisis de residuos
- Viabilidad productiva

Este enfoque garantiza una **selección objetiva, reproducible y libre de sesgos arbitrarios**.

---
