

from os import chdir

from joblib import dump, load
from numpy import number, where
from pandas import DataFrame, concat, read_csv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Función para el tratamiento de outliers por cuartiles y desviación estándar
def treat_outliers(data, numeric_columns, factor=1.5):
    treated_data = data.copy()
    for column in numeric_columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        treated_data[column] = where(
            treated_data[column] < lower_bound,
            lower_bound,
            treated_data[column])
        treated_data[column] = where(
            treated_data[column] > upper_bound,
            upper_bound,
            treated_data[column])
    return treated_data


class Classifier:
    """


    Clase para procesar datos, entrenar un modelo de RandomForest y realizar predicciones.

    Métodos
    -------
    preprocess_data():
        Método para preprocesar datos. (Personalizar según necesidad)

    transform_data():
        Método para transformar los datos (imputación, escalado, codificación, etc.)

    split_data():
        Divide los datos en conjuntos de entrenamiento y prueba.

    train_model(X_train, y_train):
        Entrena el modelo RandomForest con los datos de entrenamiento.

    evaluate_model(X_test, y_test):
        Evalúa el modelo en los datos de prueba y devuelve la precisión.

    save_model(filename):
        Guarda el modelo entrenado y los objetos de preprocesamiento en archivos.

    load_model(filename):
        Carga el modelo y los objetos de preprocesamiento desde archivos.

    retrain_model(new_data_path):
        Reentrena el modelo con nuevos datos.

    treat_outliers(data, numeric_columns, factor=1.5):
        Trata los outliers en las columnas numéricas.

    transform_for_prediction(X):
        Transforma los datos nuevos para predicción utilizando los mismos pasos de preprocesamiento.

    predict(new_data):
        Realiza predicciones sobre nuevos datos.
    """

    def __init__(self):
        """
        Inicializa la clase Classifier.

        Parameters
        ----------
        data_path : str, optional
            Ruta al archivo CSV que contiene los datos. Si no se proporciona, se debe cargar manualmente.
        """

        self.model = None
        # Convertir las columnas a su dtype correcto
        self.numeric_columns = [
            'PassengerId',
            'Pclass',
            'Age',
            'SibSp',
            'Parch',
            'Fare']
        self.categorical_columns = [
            'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        self.label = ['Survived']

        self.columns = ['PassengerId',
                        'Pclass',
                        'Name',
                        'Sex',
                        'Age',
                        'SibSp',
                        'Parch',
                        'Ticket',
                        'Fare',
                        'Cabin',
                        'Embarked']

        self.class_dict = {0: 'No sobrebvivio', 1: 'Sobrevivio'}

    def loader(self, data_path):

        self.data = read_csv(data_path)
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def preprocess_data(self):
        self.data = treat_outliers(self.data, self.numeric_columns)

    def transform_data(self):

        # Imputar datos faltantes
        self.data[self.numeric_columns] = self.numeric_imputer.fit_transform(
            self.data[self.numeric_columns])
        self.data[self.categorical_columns] = self.categorical_imputer.fit_transform(
            self.data[self.categorical_columns])

        # Escalar características numéricas
        self.data[self.numeric_columns] = self.scaler.fit_transform(
            self.data[self.numeric_columns])

        # Codificar características categóricas
        encoded_categories = self.encoder.fit_transform(
            self.data[self.categorical_columns]).toarray()
        encoded_df = DataFrame(
            encoded_categories,
            columns=self.encoder.get_feature_names_out(
                self.categorical_columns))

        numeric_columns_extend = self.numeric_columns + ['Survived']
        # Concatenar datos transformados
        self.data = concat(
            [self.data[numeric_columns_extend], encoded_df], axis=1)

    def split_data(self):
        X = self.data.drop(columns=['Survived'])
        y = self.data['Survived']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def save_model(self, filename):

        filename = 'model/' + filename

        dump(self.model, filename)

        dump(self.numeric_imputer, filename + '_numeric_imputer.joblib')
        dump(
            self.categorical_imputer,
            filename +
            '_categorical_imputer.joblib')
        dump(self.scaler, filename + '_scaler.joblib')
        dump(self.encoder, filename + '_encoder.joblib')

    def load_model(self, filename):
        self.model = load(filename)
        self.numeric_imputer = load(filename + '_numeric_imputer.joblib')
        self.categorical_imputer = load(
            filename + '_categorical_imputer.joblib')
        self.scaler = load(filename + '_scaler.joblib')
        self.encoder = load(filename + '_encoder.joblib')

    def transform_for_prediction(self, X):
        # Tratamiento de outliers
        X = treat_outliers(X, self.numeric_columns)

        # Imputar valores faltantes
        X[self.numeric_columns] = self.numeric_imputer.transform(
            X[self.numeric_columns])
        X[self.categorical_columns] = self.categorical_imputer.transform(
            X[self.categorical_columns])

        # Escalar características numéricas
        X[self.numeric_columns] = self.scaler.transform(
            X[self.numeric_columns])

        # One-Hot Encoding para características categóricas
        encoded_cats = self.encoder.transform(
            X[self.categorical_columns]).toarray()
        encoded_cat_df = DataFrame(
            encoded_cats,
            columns=self.encoder.get_feature_names_out(
                self.categorical_columns))

        X = X.drop(columns=self.categorical_columns).reset_index(drop=True)
        X = concat([X, encoded_cat_df], axis=1)

        return X

    def predict(self, new_data):

        # Asegurarse de que los nuevos datos sean un DataFrame de pandas
        if not isinstance(new_data, DataFrame):
            new_data = DataFrame(new_data, columns=self.columns)

        # Transformar los datos nuevos para predicción
        X_new = self.transform_for_prediction(new_data)

        # Hacer predicciones
        predictions = self.model.predict(X_new)

        """
        predictions_proba = self.model.predict_proba(X_new)
        predicts_ = []
        for prediction in predictions:
            predicts_.append(self.class_dict[prediction])

        """

        return predictions
