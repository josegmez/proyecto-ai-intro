import pickle
from os import makedirs
from os.path import exists

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from model.models import PredictionData
from model.utils import cleaning_data


class Model:
    model = None
    preprocessing = None
    model_path = "../models/model.pkl"
    training_data_path = "../data/train.csv"
    preprocessing_path = "../models/preprocessing.pkl"
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

    def __init__(self):
        if not self.model:
            self.model = self.load_model(self.model_path)
        if not self.preprocessing:
            self.preprocessing = self.load_model(self.preprocessing_path)

    def train(self) -> float:
        # Cargar y limpiar los datos de entrenamiento
        df_train = pd.read_csv(self.training_data_path, index_col="id")
        df_train_cleaned = cleaning_data(df_train)

        # Preprocesamiento de los datos
        num_pipeline = Pipeline([("scaler", StandardScaler())])
        cat_pipeline = Pipeline(
            [
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                )
            ]
        )

        preprocessing = ColumnTransformer(
            [
                ("num", num_pipeline, self.num_features),
                ("cat", cat_pipeline, self.cat_features),
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
            iterations=1000,
            depth=10,
            random_strength=0,
            l2_leaf_reg=0.3,
            task_type="CPU",
            random_seed=42,
            verbose=500,
        )
        model.fit(X_train_pool)

        if not exists("../models"):
            makedirs("../models")

        # Guardar el modelo entrenado
        with open(self.model_path, "wb") as file:
            pickle.dump(model, file)

        # Guardar el preprocesamiento de los datos
        with open(self.preprocessing_path, "wb") as file:
            pickle.dump(preprocessing, file)

        return model.score(X_train_pool)

    def predict(self, data: PredictionData) -> int:
        df_predict = pd.DataFrame(data, index=[0])
        df_predict_cleaned = cleaning_data(df_predict)

        # # Asegurarse de que las columnas categóricas estén en el formato correcto
        for cat_col in self.cat_features:
            df_predict_cleaned[cat_col] = df_predict_cleaned[cat_col].astype(str)

        # # Transformar el conjunto de prueba
        x = self.preprocessing.transform(df_predict_cleaned)
        pool = Pool(x)

        if not self.model:
            raise ValueError("El modelo no ha sido cargado.")

        return self.model.predict(pool)

    def load_model(self, model_path):
        if not exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo '{model_path}'.")

        print(f"Cargando modelo desde '{model_path}'...")

        with open(model_path, "rb") as file:
            model = pickle.load(file)

            return model

    def close(self):
        self.model = None
        self.preprocessing = None
