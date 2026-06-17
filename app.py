from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelo entrenado
modelo = joblib.load("modelo_regresion.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None

    if request.method == "POST":
        hours_studied = float(request.form["hours_studied"])
        previous_scores = float(request.form["previous_scores"])
        extracurricular = int(request.form["extracurricular"])
        sleep_hours = float(request.form["sleep_hours"])
        sample_papers = float(request.form["sample_papers"])

        datos = pd.DataFrame([{
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": sample_papers
        }])

        resultado = modelo.predict(datos)
        prediccion = round(resultado[0], 2)

    return render_template("index.html", prediccion=prediccion)

if __name__ == "__main__":
    app.run(debug=True)