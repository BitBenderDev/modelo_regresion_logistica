import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

        # DataFrame limpio
        self.df_clean = None

        # Columnas categóricas
        self.categorical_columns = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
        self.label_encoders = {}
        self.categorias_originales = {}

        # Info de features
        self.feature_names = None
        self.numeric_columns = None
        self.scaler = None

        # Datos de train/test escalados
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def prepare(self):
        # 1. Cargar datos
        df = pd.read_csv(self.csv_path)

        # 2. Copia y limpieza básica
        df_clean = df.copy()

        # Edad máxima 100
        df_clean = df_clean[df_clean["person_age"] <= 100]

        # 3. Quitar outliers fuertes de income (IQR)
        Q1 = df_clean["person_income"].quantile(0.25)
        Q3 = df_clean["person_income"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[
            (df_clean["person_income"] >= lower_bound)
            & (df_clean["person_income"] <= upper_bound)
        ]

        # 4. Imputar nulos en columnas numéricas específicas
        imputer_num = SimpleImputer(strategy="median")
        numeric_cols_with_na = ["person_emp_length", "loan_int_rate"]
        df_clean[numeric_cols_with_na] = imputer_num.fit_transform(
            df_clean[numeric_cols_with_na]
        )

        # 5. Nuevas características
        df_clean["debt_to_income_ratio"] = (
            df_clean["loan_amnt"] / df_clean["person_income"]
        )
        df_clean["income_to_loan_ratio"] = (
            df_clean["person_income"] / df_clean["loan_amnt"]
        )
        df_clean["is_employed"] = (df_clean["person_emp_length"] > 0).astype(int)

        # 6. Codificación de categóricas
        for col in self.categorical_columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le
            self.categorias_originales[col] = list(le.classes_)

        # 7. Separar X e y
        X = df_clean.drop("loan_status", axis=1)
        y = df_clean["loan_status"]

        # 8. Train / test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # 9. Escalado
        self.scaler = StandardScaler()
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[self.numeric_columns] = self.scaler.fit_transform(
            X_train[self.numeric_columns]
        )
        X_test_scaled[self.numeric_columns] = self.scaler.transform(
            X_test[self.numeric_columns]
        )

        # Guardar en atributos
        self.df_clean = df_clean
        self.feature_names = list(X.columns)
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

    def construir_entrada_escalada(
        self,
        age,
        income,
        emp_length,
        loan_amnt,
        loan_int_rate,
        home_ownership,
        loan_intent,
        loan_grade,
        default_on_file,
    ):
        # Crear diccionario base con todas las columnas
        data = {col: 0 for col in self.feature_names}

        # Variables numéricas principales
        data["person_age"] = age
        data["person_income"] = income
        data["person_emp_length"] = emp_length
        data["loan_amnt"] = loan_amnt
        data["loan_int_rate"] = loan_int_rate

        # Categóricas codificadas
        data["person_home_ownership"] = self.label_encoders[
            "person_home_ownership"
        ].transform([home_ownership])[0]

        data["loan_intent"] = self.label_encoders["loan_intent"].transform(
            [loan_intent]
        )[0]

        data["loan_grade"] = self.label_encoders["loan_grade"].transform(
            [loan_grade]
        )[0]

        data["cb_person_default_on_file"] = self.label_encoders[
            "cb_person_default_on_file"
        ].transform([default_on_file])[0]

        # Nuevas características
        data["debt_to_income_ratio"] = loan_amnt / income
        data["income_to_loan_ratio"] = income / loan_amnt
        data["is_employed"] = 1 if emp_length > 0 else 0

        df_nuevo = pd.DataFrame([data])
        df_nuevo_scaled = df_nuevo.copy()
        df_nuevo_scaled[self.numeric_columns] = self.scaler.transform(
            df_nuevo[self.numeric_columns]
        )

        return df_nuevo_scaled

