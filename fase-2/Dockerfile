# Seleccionamos la imagen base de Python 3.9
FROM python:3.9-slim

# Seteamos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos del proyecto a la carpeta de trabajo
COPY ./fase-2 /app/fase-2
COPY ./data /app/data
COPY ./models /app/models

# Instalamos las dependencias
RUN pip install --no-cache-dir -r /app/fase-2/requirements.txt

# Ejecutamos el script de entrenamiento
RUN python /app/fase-2/train.py --data_file /app/data/train.csv --model_file /app/models/model.pkl --preprocessing_file /app/models/preprocessing.pkl

# Usamos un entrypoint para poder pasar comandos al contenedor
ENTRYPOINT ["python"]

# El comando por defecto será la predicción, pero lo puedes cambiar al ejecutar
CMD ["/app/fase-2/predict.py", "--input_file", "/app/data/test.csv", "--model_file", "/app/models/model.pkl", "--preprocessing_file", "/app/models/preprocessing.pkl", "--predictions_file", "/app/data/predicts.csv"]