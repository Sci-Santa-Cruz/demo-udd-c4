
from classifier import Classifier


path = 'data/raw/titanic.csv'

# Ejemplo de uso
data_processor = Classifier()
data_processor.loader(path)
data_processor.preprocess_data()
data_processor.transform_data()
X_train, X_test, y_train, y_test = data_processor.split_data()
data_processor.train_model(X_train, y_train)
# Suponiendo que tienes nuevos datos en un archivo llamado 'nuevos_datos.csv'
accuracy = data_processor.evaluate_model(X_test, y_test)
print("Accuracy:", accuracy)
salida  = data_processor.save_model('titanic_model')