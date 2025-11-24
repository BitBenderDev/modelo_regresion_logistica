import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_preprocessor import DataPreprocessor


class LogisticCreditModel:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        # métricas del modelo
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

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

        # métricas
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, zero_division=0)
        self.recall = recall_score(y_test, y_pred, zero_division=0)
        self.f1 = f1_score(y_test, y_pred, zero_division=0)

    def predecir(self, df_nuevo_scaled: pd.DataFrame):
        """
        Recibe un DataFrame con una fila ya escalada
        y devuelve (probabilidad_clase_1, predicción_0_1)
        """
        prob = self.model.predict_proba(df_nuevo_scaled)[0][1]
        pred = self.model.predict(df_nuevo_scaled)[0]
        return prob, pred
