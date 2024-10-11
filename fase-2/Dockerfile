# Seleccionamos la imagen base de Python 3.9
FROM python:3.9-slim

# Seteamos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos de la carpeta actual a la carpeta de trabajo
COPY ./fase-2 .

COPY ../data ../data

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt


# Ejecutamos el script de entrenamiento
RUN python train.py

# Ejecutamos el script de predicción
CMD ["python", "predict.py"]