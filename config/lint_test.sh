#!/bin/bash

# Este script realiza el análisis estático y formateo de código en archivos Python en un directorio.
# Modo de uso: sh lint_script.sh <directorio_con_archivos_py>

# Verificar si se proporciona un argumento
if [ $# -eq 0 ]; then
    echo "Uso: $0 <directorio_con_archivos_py>"
    exit 1
fi

# Obtener el directorio con archivos Python desde el primer argumento
directory=$1

echo "Iniciando análisis estático y formateo de código..."

# Formateo de código con autopep8 para todos los archivos .py en el directorio
echo "Formateo de código con autopep8..."
find "$directory" -name "*.py" -exec python -m autopep8 --in-place --aggressive --aggressive {} +
echo "Formateo de código con autopep8 completado."

# Organización de importaciones con isort para todos los archivos .py en el directorio
echo "Organización de importaciones con isort..."
find "$directory" -name "*.py" -exec python -m isort {} +
echo "Organización de importaciones con isort completada."

# Verificación del código con flake8 para todos los archivos .py en el directorio
echo "Verificación del código con flake8..."
find "$directory" -name "*.py" -exec python -m flake8 {} +
echo "Verificación del código con flake8 completada."

# Análisis estático con pylint para todos los archivos .py en el directorio
echo "Análisis estático con pylint..."
find "$directory" -name "*.py" -exec python -m pylint {} +
echo "Análisis estático con pylint completado."

# Verificación del formato de código con black para todos los archivos .py en el directorio
echo "Verificación del formato de código con black..."
find "$directory" -name "*.py" -exec python -m black --check {} +
echo "Verificación del formato de código con black completada."

# Escaneo de seguridad con bandit para todos los archivos .py en el directorio
echo "Escaneo de seguridad con bandit..."
find "$directory" -name "*.py" -exec python -m bandit -r {} +
echo "Escaneo de seguridad con bandit completado."

# Verificación de tipos con mypy para todos los archivos .py en el directorio
echo "Verificación de tipos con mypy..."
find "$directory" -name "*.py" -exec python -m mypy {} +
echo "Verificación de tipos con mypy completada."

echo "Análisis estático y formateo de código completado."
