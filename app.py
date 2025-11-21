import streamlit as st

from data_preprocessor import DataPreprocessor
from logistic_model import LogisticCreditModel


class CreditRiskApp:
    def __init__(self, preprocessor: DataPreprocessor, model: LogisticCreditModel):
        self.prep = preprocessor
        self.model = model

    def run(self):
        st.title("Sistema de Regresión Logística para Riesgo de Crédito")

        st.sidebar.header("Información del modelo")
        st.sidebar.write(f"Accuracy en test: **{self.model.accuracy:.3f}**")
        st.sidebar.write("Algoritmo: **Regresión Logística**")

        st.write(
            "Ingresa los datos del solicitante para predecir `loan_status` "
            "(0 = bajo riesgo, 1 = alto riesgo)."
        )

        age = st.number_input("Edad (person_age)", min_value=18, max_value=100, value=30)
        income = st.number_input(
            "Ingreso anual (person_income)",
            min_value=1000,
            max_value=1_000_000,
            value=50_000,
            step=1000,
        )
        emp_length = st.number_input(
            "Años de experiencia laboral (person_emp_length)",
            min_value=0,
            max_value=60,
            value=5,
        )
        loan_amnt = st.number_input(
            "Monto del préstamo (loan_amnt)",
            min_value=500,
            max_value=100_000,
            value=10_000,
            step=500,
        )
        loan_int_rate = st.number_input(
            "Tasa de interés del préstamo (loan_int_rate)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
        )

        home_ownership = st.selectbox(
            "Tipo de vivienda (person_home_ownership)",
            self.prep.categorias_originales["person_home_ownership"],
        )
        loan_intent = st.selectbox(
            "Intención del préstamo (loan_intent)",
            self.prep.categorias_originales["loan_intent"],
        )
        loan_grade = st.selectbox(
            "Grado del préstamo (loan_grade)",
            self.prep.categorias_originales["loan_grade"],
        )
        default_on_file = st.selectbox(
            "Default previo en archivo (cb_person_default_on_file)",
            self.prep.categorias_originales["cb_person_default_on_file"],
        )

        if st.button("Predecir"):
            df_nuevo_scaled = self.prep.construir_entrada_escalada(
                age,
                income,
                emp_length,
                loan_amnt,
                loan_int_rate,
                home_ownership,
                loan_intent,
                loan_grade,
                default_on_file,
            )

            prob, pred = self.model.predecir(df_nuevo_scaled)

            st.write(
                f"Probabilidad de `loan_status = 1` (alto riesgo): **{prob:.3f}**"
            )

            if pred == 1:
                st.error("Predicción: ALTO RIESGO (loan_status = 1)")
            else:
                st.success("Predicción: BAJO RIESGO (loan_status = 0)")



def construir_app():
    preprocessor = DataPreprocessor("credit_risk_dataset.csv")
    preprocessor.prepare()

    model = LogisticCreditModel(preprocessor)
    model.train()

    app = CreditRiskApp(preprocessor, model)
    return app


def main():
    app = construir_app()
    app.run()


if __name__ == "__main__":
    main()

