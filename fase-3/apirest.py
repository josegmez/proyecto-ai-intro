from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from model import Model
from model.models import ModelPrediction, PredictionData


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargamos el modelo al inicio de la aplicación
    Model()
    yield
    # Cerramos el modelo al finalizar la aplicación
    Model().close()


app = FastAPI(lifespan=lifespan)


@app.post("/train")
def train_handler(model=Depends(Model)):
    try:
        score = model.train()
        return f"Modelo entrenado con un score de {score}"
    except Exception:
        return "Error al entrenar el modelo"


@app.post("/predict")
def predict_handler(data: PredictionData, model=Depends(Model)) -> ModelPrediction:
    prediction = model.predict(data=data.model_dump())
    parsed_price = int(prediction[0])
    return ModelPrediction(price=parsed_price)
