#!/bin/bash

set -e  # Detener la ejecución en caso de error

# Instalar nano
sudo apt-get update
sudo apt-get install -y nano

# Actualizar pip
pip install --upgrade pip

pip uninstall -y tomli typing_extensions

# Instalar bibliotecas necesarias para los tests
pip install tomli typing_extensions pip-audit flake8==4.0.1 pylint==2.12.2 black==22.3.0 bandit==1.7.1 mypy==0.931 autopep8==1.5.7 nbqa

# Actualizar bibliotecas
pip install --upgrade flake8 pycodestyle


echo "Configuración de pre-commit y bibliotecas instaladas correctamente."