import os
import pickle

import pandas as pd
from catboost import Pool
from sklearn.impute import SimpleImputer


# Función para limpiar los datos
def cleaning_data(df):
    """
    This function cleans the data by filling the missing values and removing the outliers.

    Args:
    df : DataFrame : The dataset to be cleaned.
    """

    fuel_type_imputer = SimpleImputer(strategy="most_frequent")
    df["fuel_type"] = fuel_type_imputer.fit_transform(df[["fuel_type"]]).ravel()

    missing_label_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    df[["accident", "clean_title"]] = missing_label_imputer.fit_transform(
        df[["accident", "clean_title"]]
    )

    return df


# Definición de características
num_features = ["milage"]
cat_features = [
    "clean_title",
    "accident",
    "model",
    "transmission",
    "engine",
    "ext_col",
    "fuel_type",
    "int_col",
    "brand",
    "model_year",
]
if not os.path.exists("../data/test.csv"):
    raise FileNotFoundError("No se encontró el archivo '../data/test.csv'.")

# Cargar y limpiar los datos de prueba
df_test = pd.read_csv("../data/test.csv", index_col="id")
df_test_cleaned = cleaning_data(df_test)

# Asegurarse de que las columnas categóricas estén en el formato correcto
for cat_col in cat_features:
    df_test_cleaned[cat_col] = df_test_cleaned[cat_col].astype(str)

# Transformar el conjunto de prueba
X_test = df_test_cleaned

if not os.path.exists("../models/preprocessing.pkl"):
    raise FileNotFoundError("No se encontró el archivo '../models/preprocessing.pkl'.")

with open("../models/preprocessing.pkl", "rb") as file:
    preprocessing = pickle.load(file)

X_test_transformed = preprocessing.transform(X_test)

# Crear el Pool para las predicciones
X_test_pool = Pool(X_test_transformed)


# Cargar el modelo entrenado
modelPath = "../models/model.pkl"

if not os.path.exists(modelPath):
    raise FileNotFoundError(f"No se encontró el archivo '{modelPath}'.")

with open(modelPath, "rb") as file:
    model = pickle.load(file)

# Realizar predicciones
predictions = model.predict(X_test_pool)

# Guardar las predicciones en un archivo CSV
output = pd.DataFrame({"id": df_test_cleaned.index, "predicted_price": predictions})
output.to_csv("../data/predicts.csv", index=False)

print("Predicciones guardadas en '../data/predicts.csv'.")
