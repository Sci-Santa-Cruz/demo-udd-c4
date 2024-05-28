## Proyecto 7: Titanic  

### Descripción
Este proyecto utiliza el famoso dataset del Titanic para entrenar un modelo de clasificación que predice la supervivencia de los pasajeros. El modelo y el procesamiento de datos se implementan en un entorno Flask para proporcionar una interfaz web interactiva.

### Tabla de Contenidos
1. [Instalación](#instalación)
2. [Uso](#uso)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Contribución](#contribución)
5. [Autores](#autores)
6. [Licencia](#licencia)
7. [Agradecimientos](#agradecimientos)

### Instalación

1. **Clonar Repositorio**
   - Clona este repositorio en tu máquina local:
     ```
     git clone https://github.com/tu_usuario/proyecto.git
     ```

2. **Crear Entorno Virtual**
   - Crea un entorno virtual llamado `predict_survived` usando la herramienta que prefieras, como `virtualenv` o `conda`:
     ```
      virtualenv -p python3.10 predict_survived
     ```

3. **Instalar Requisitos**
   - Activa el entorno virtual:
     ```
     source predict_survived/bin/activate   # Para Unix/Linux
     predict_survived\Scripts\activate       # Para Windows
     ```
   - Instala los paquetes requeridos listados en `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

4. **Descargar Datos de Entrenamiento**
   - Descarga el archivo de datos de entrenamiento desde [link_to_data](https://example.com/data.csv) y colócalo en el directorio `data/raw` con el nombre titanic.csv.

5. **Entrenar Clasificador**
   - Ejecuta el script `train.py` en el directorio `categorizador` para entrenar el clasificador y generar los artefactos del modelo:
     ```
     python categorizador/train.py
     ```

6. **Ejecutar API**
   - Ejecuta el script `api.py` para iniciar el servidor de la API:
     ```
     python api.py
     ```

7. **Hacer Predicciones**
   - Una vez que el servidor de la API esté en funcionamiento, navega a `http://localhost:8080/predict` en tu navegador web o utiliza una herramienta como `curl` para hacer predicciones.
    ```

### Uso
Una vez que la aplicación esté en funcionamiento, puedes acceder a ella desde tu navegador web visitando la dirección `http://localhost:5000`. La aplicación te proporcionará una interfaz interactiva donde podrás ingresar datos de pasajeros del Titanic y obtener predicciones sobre su supervivencia.

### Estructura del Proyecto
La estructura del proyecto es la siguiente:

- `app/`: Contiene los archivos de la aplicación Flask.
- `data/`: Directorio para almacenar los datos del dataset Titanic.
- `model/`: Contiene el modelo de clasificación entrenado.
- `templates/`: Plantillas HTML para la interfaz web de la aplicación.
- `static/`: Archivos estáticos como CSS, JavaScript e imágenes.
- `requirements.txt`: Lista de dependencias del proyecto.

### Contribución
Si deseas contribuir al proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-caracteristica`).
3. Realiza tus cambios y haz commit (`git commit -am 'Agrega nueva característica'`).
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`).
5. Crea un nuevo Pull Request.

### Autores
- [Tu Nombre](https://github.com/tu_usuario)

### Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

### Agradecimientos
- A todos los contribuyentes que han ayudado en el desarrollo de este proyecto.