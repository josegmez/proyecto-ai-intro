FROM python:3.10-slim

WORKDIR /app

# Copiamos los archivos del proyecto a la carpeta de trabajo
COPY ./fase-3 /fase-3
COPY ./data /data
COPY ./models /models

RUN pip install --no-cache-dir -r /fase-3/requirements.txt

# Exponemos el puerto 8000
EXPOSE 8000

CMD ["fastapi", "run", "/fase-3/apirest.py"]