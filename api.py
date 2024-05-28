from categorizador.classifier import Classifier
from flask import Flask, jsonify, request
from pandas import DataFrame
# Importa la clase Classifier de tu módulo

app = Flask(__name__)

# Cargar la instancia de Classifier con el modelo entrenado
data_processor = Classifier()
data_processor.load_model('model/titanic_model')

# Ruta para realizar predicciones


@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de form-data de la solicitud
    features = {
        "PassengerId": int(request.form['PassengerId']),
        "Pclass": int(request.form['Pclass']),
        "Name": request.form['Name'],
        "Sex": request.form['Sex'],
        "Age": float(request.form['Age']),
        "SibSp": int(request.form['SibSp']),
        "Parch": int(request.form['Parch']),
        "Ticket": request.form['Ticket'],
        "Fare": float(request.form['Fare']),
        "Cabin": request.form['Cabin'],
        "Embarked": request.form['Embarked']
    }

    # Crear un DataFrame a partir de los datos recibidos
    df = DataFrame([features])

    # Realizar la predicción con el modelo cargado
    predictions = data_processor.predict(df)

    print(predictions)
    # Devolver la respuesta en formato JSON
    return jsonify({"response": predictions.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
