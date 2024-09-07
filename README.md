# Proyecto Sustituto - Introdución a la Inteligencia Artificial

## Integrantes

- **1041410002** - Jose David Gómez Muñetón
  - Ingeniería de Sistemas
- **1001477904** - Juan Pablo Arango Gaviria
  - Ingeniería de Sistemas

## Ejecución

### Fase 1

> [!NOTE]
> La tarea de entrenamiento se puede hacer con tareas en CPU o GPU, para cambiar el dispositivo de entrenamiento se debe modificar la variable `task_type` en el notebook `/fase-1/model.ipynb` en la última celda de la sección #2. En caso de querer entrenar en GPU, se debe cambiar el valor de la variable a `GPU` y debe tener los drivers de CUDA instalados en su máquina.

Para ejecutar la fase 1 del proyecto, se deben ejecutar todas las celdas del notebook `fase-1/model.ipynb`.
Este notebook se encarga de descargar las librerías necesarias, cargar los datos, preprocesarlos, entrenar el modelo con los datos de `/data/train.csv`, evaluarlo y generar las predicciones para el conjunto de prueba (`/data/train.csv`).
