import os
import pickle
import argparse

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Argumentos
parser = argparse.ArgumentParser(description='Entrena un modelo de predicción.')
parser.add_argument('--data_file', required=True, type=str, help='Archivo CSV de entrenamiento')
parser.add_argument('--model_file', required=True, type=str, help='Archivo donde se guardará el modelo entrenado')
parser.add_argument('--preprocessing_file', required=True, type=str, help='Archivo donde se guardará el preprocesamiento')
args = parser.parse_args()

data_file = args.data_file
model_file = args.model_file
preprocessing_file = args.preprocessing_file

rs = 42
task_type = "CPU"


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

# Cargar y limpiar los datos de entrenamiento
df_train = pd.read_csv(data_file, index_col="id")
df_train_cleaned = cleaning_data(df_train)

# Preprocesamiento de los datos
num_pipeline = Pipeline([("scaler", StandardScaler())])
cat_pipeline = Pipeline(
    [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
)

preprocessing = ColumnTransformer(
    [
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ]
)

X = df_train_cleaned.drop(columns=["price"])
y = df_train_cleaned["price"]

# Ajustar y transformar el conjunto de entrenamiento
X_train_transformed = preprocessing.fit_transform(X)

# Crear el Pool para el conjunto de entrenamiento
X_train_pool = Pool(X_train_transformed, y)

# Entrenar el modelo
model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    learning_rate=0.042,
    iterations=5000,
    depth=10,
    random_strength=0,
    l2_leaf_reg=0.3,
    task_type=task_type,
    random_seed=rs,
    verbose=500,
)
model.fit(X_train_pool)

print("Modelo entrenado exitosamente.")
print(model.score(X_train_pool))

if not os.path.exists("/app/models"):
    os.makedirs("../models")

# Guardar el modelo entrenado
with open(model_file, "wb") as file:
    pickle.dump(model, file)

# Guardar el preprocesamiento de los datos
with open(preprocessing_file, "wb") as file:
    pickle.dump(preprocessing, file)
