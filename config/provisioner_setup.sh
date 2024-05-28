#!/bin/bash

# Paso 1: Clonar Repositorio
git clone https://github.com/tu_usuario/proyecto.git
cd proyecto

# Paso 2: Crear Entorno Virtual
virtualenv -p python3.10 predict_survived

# Paso 3: Instalar Requisitos
source predict_survived/bin/activate   # Para Unix/Linux
# predict_survived\Scripts\activate     # Para Windows
pip install -r requirements.txt

# Paso 4: Validar Existencia de Directorio y Descargar Datos de Entrenamiento
mkdir -p data/raw
if [ ! -d "data/raw" ]; then
    echo "El directorio data/raw no existe. Por favor, crea el directorio manualmente y ejecuta este script nuevamente."
    exit 1
fi

# Descargar el archivo solo si no existe
if [ ! -f "data/raw/titanic.csv" ]; then
    curl -o data/raw/titanic.csv https://link_a_los_datos/titanic.csv
else
    echo "El archivo 'titanic.csv' ya existe en el directorio 'data/raw'."
fi

# Paso 5: Entrenar Clasificador
cd categorizador
python train.py
cd ..

# Paso 6: Ejecutar API
python api.py

# Paso 7: Hacer Predicciones
# Una vez que el servidor de la API esté en funcionamiento, navegar a `http://localhost:8080/predict` en el navegador web o usar `curl` para hacer predicciones.
