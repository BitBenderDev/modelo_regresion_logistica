import streamlit as st

from data_preprocessor import DataPreprocessor
from logistic_model import LogisticCreditModel

# Configuración general de la página (título, ícono, ancho)
st.set_page_config(
    page_title="Predicción de Riesgo de Crédito",
    page_icon="💳",
    layout="wide",
)


class CreditRiskApp:
    def __init__(self, preprocessor: DataPreprocessor, model: LogisticCreditModel):
        self.prep = preprocessor
        self.model = model

    def run(self):
        # Título principal
        st.title("🔍 Sistema de Predicción de Riesgo de Crédito")
        st.markdown(
            "Modelo de **Regresión Logística** sobre el dataset de riesgo de crédito."
        )
        st.markdown("---")

        # Layout: columna izquierda -> métricas, derecha -> formulario y resultado
        col_metrics, col_form = st.columns([1, 2])

        # ============ COLUMNA IZQUIERDA: MÉTRICAS ============
        with col_metrics:
            st.markdown("### 📊 Métricas del modelo")

            st.metric("Accuracy", f"{self.model.accuracy * 100:.2f} %")
            st.metric("Precisión", f"{self.model.precision * 100:.2f} %")
            st.metric("Recall", f"{self.model.recall * 100:.2f} %")
            st.metric("F1-Score", f"{self.model.f1 * 100:.2f} %")

            st.markdown("---")
            st.caption(
                "Las métricas se calculan sobre el conjunto de prueba después del entrenamiento."
            )

        # ============ COLUMNA DERECHA: FORMULARIO Y PREDICCIÓN ============
        with col_form:
            st.markdown("### 🧾 Datos del solicitante")

            # Sub-columnas para organizar mejor el formulario
            c1, c2 = st.columns(2)

            with c1:
                age = st.number_input(
                    "Edad (person_age)", min_value=18, max_value=100, value=30
                )
                income = st.number_input(
                    "Ingreso anual (person_income)",
                    min_value=1_000,
                    max_value=1_000_000,
                    value=50_000,
                    step=1_000,
                )
                emp_length = st.number_input(
                    "Años de experiencia laboral (person_emp_length)",
                    min_value=0,
                    max_value=60,
                    value=5,
                )

            with c2:
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

            # Otra fila de selects
            c3, c4 = st.columns(2)
            with c3:
                loan_intent = st.selectbox(
                    "Intención del préstamo (loan_intent)",
                    self.prep.categorias_originales["loan_intent"],
                )
            with c4:
                loan_grade = st.selectbox(
                    "Grado del préstamo (loan_grade)",
                    self.prep.categorias_originales["loan_grade"],
                )

            default_on_file = st.selectbox(
                "Default previo en archivo (cb_person_default_on_file)",
                self.prep.categorias_originales["cb_person_default_on_file"],
            )

            st.markdown("---")

            # Botón para predecir
            if st.button("🚨 Realizar predicción"):
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

                # Mostrar resultado en una "tarjeta"
                st.markdown("### Resultado de la predicción")

                prob_text = f"Probabilidad de `loan_status = 1` (ALTO riesgo): **{prob:.3f}**"

                if pred == 1:
                    st.error(
                        f"{prob_text}\n\n"
                        "💥 **Predicción: ALTO RIESGO** de incumplimiento (loan_status = 1)."
                    )
                else:
                    st.success(
                        f"{prob_text}\n\n"
                        "✅ **Predicción: BAJO RIESGO** de incumplimiento (loan_status = 0)."
                    )

            st.markdown("---")
            st.caption(
                "Sistema de Predicción de Riesgo de Crédito | Desarrollado con Streamlit"
            )


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

