#app.py
from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import skfuzzy as fuzz  # Para FCM

app = Flask(__name__)
app.secret_key = "tu_clave_secreta"  # Reemplaza con una clave segura

# Cargar los modelos entrenados
logistic_model = joblib.load("models/logistic_regression.pkl")
svm_model = joblib.load("models/svm.pkl")
fcm_model = joblib.load("models/fcm.pkl")

# ------------------------------
# RUTA DE EVALUACIÓN INDIVIDUAL
# ------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, prob=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Capturar valores de C1 a C30
        values = []
        for i in range(1, 31):
            key = f"C{i}"
            value = request.form.get(key, "").strip()
            if not value:
                raise ValueError(f"Falta el dato en el campo {key}")
            try:
                num = float(value)
            except ValueError:
                raise ValueError(f"El dato en {key} debe ser numérico")
            values.append(num)
        
        input_array = np.array([values])  # Forma (1, 30)
        
        # Capturar el modelo seleccionado
        model_selection = request.form.get("selected_model")
        if not model_selection:
            raise ValueError("Debe seleccionar un modelo")

        # Obtener predicción según el modelo elegido
        if model_selection == "logistic":
            y_pred = logistic_model.predict(input_array)
        elif model_selection == "svm":
            y_pred = svm_model.predict(input_array)
        elif model_selection == "fcm":
            X_T = input_array.T  # ahora (30,1)
           
            u_new, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X_T, 
                fcm_model["centroids"], 
                fcm_model["m"], 
                error=fcm_model["error"], 
                maxiter=fcm_model["maxiter"]
            )
            # Para la única muestra, obtener el cluster con mayor membresía
            predicted_cluster = np.argmax(u_new, axis=0)[0]
            # Asignar la etiqueta según el cluster (la lista cluster_labels se entrenó previamente)
            y_pred = fcm_model["cluster_labels"][predicted_cluster]
        else:
            raise ValueError("Modelo seleccionado inválido")
        
        result = "Presenta FGR" if y_pred == 1 else "Estado Normal"
        return render_template("index.html", result=result)
    except Exception as e:
        flash(str(e))
        return redirect(url_for("index"))

# ------------------------------
# RUTA DE EVALUACIÓN POR LOTES
# ------------------------------
@app.route("/batch", methods=["GET"])
def batch_page():
    return render_template("lotes.html")

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        # Capturar modelo seleccionado
        model_selection = request.form.get("selected_model")
        if not model_selection:
            raise ValueError("Debe seleccionar un modelo para la evaluación por lotes.")

        # Capturar el archivo subido
        if "file" not in request.files:
            raise ValueError("No se encontró ningún archivo.")
        file = request.files["file"]
        if file.filename == "":
            raise ValueError("No se seleccionó ningún archivo.")

        # Leer archivo Excel subido
        df = pd.read_excel(file)
        expected_cols = [f"C{i}" for i in range(1, 31)]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"El archivo debe contener la columna: {col}")
        
        X = df[expected_cols].values  # Forma (n_samples, 30)

        # Revisar si existe la columna C31 (etiqueta real)
        y_true = df["C31"].values if "C31" in df.columns else None

        # Obtener predicciones según el modelo seleccionado
        if model_selection == "logistic":
            y_pred = logistic_model.predict(X)
        elif model_selection == "svm":
            y_pred = svm_model.predict(X)
        elif model_selection == "fcm":
            X_T = X.T
            u_new, _, _, _, _, _ = fuzz.cluster.cmeans_predict(X_T, fcm_model["centroids"], fcm_model["m"], error=fcm_model["error"], maxiter=fcm_model["maxiter"])
            predicted_clusters = np.argmax(u_new, axis=0)
            y_pred = [fcm_model["cluster_labels"][cluster] for cluster in predicted_clusters]
        else:
            raise ValueError("Modelo seleccionado inválido.")

        # Si se dispone de etiquetas verdaderas, calcular métricas
        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            return render_template("lotes_result.html", accuracy=round(acc, 3), cm=cm, model_name=model_selection)
        else:
            records = df.to_dict(orient="records")
            return render_template("lotes_result.html", predictions=y_pred, data=records, model_name=model_selection)

    except Exception as e:
        flash(str(e))
        return redirect(url_for("batch_page"))

if __name__ == "__main__":
    app.run(debug=True)
