import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_preprocessor import DataPreprocessor


class LogisticCreditModel:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.accuracy = None

    def train(self):
        # Datos de entrenamiento ya escalados
        X_train = self.preprocessor.X_train_scaled
        y_train = self.preprocessor.y_train

        # Entrenar regresión logística
        self.model.fit(X_train, y_train)

        # Evaluar en test
        X_test = self.preprocessor.X_test_scaled
        y_test = self.preprocessor.y_test

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

    def predecir(self, df_nuevo_scaled: pd.DataFrame):
        """
        Recibe un DataFrame con una fila ya escalada
        y devuelve (probabilidad_clase_1, predicción_0_1)
        """
        prob = self.model.predict_proba(df_nuevo_scaled)[0][1]
        pred = self.model.predict(df_nuevo_scaled)[0]
        return prob, pred
