#train_models.py
import pandas as pd
import joblib
import numpy as np
import skfuzzy as fuzz
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Cargar dataset ---
df = pd.read_excel("FGR_dataset.xlsx")  

# --- Separar características (C1 a C30) y target (C31) ---
X = df.drop(columns=["C31"])  # Características
y = df["C31"]  # Variable objetivo (0 = Estado Normal, 1 = FGR Positivo)

# --- Aplicar SMOTE para balancear el dataset si está desbalanceado ---
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# --- Dividir en conjunto de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# --- Normalizar los datos para mejorar el rendimiento de los modelos ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------------------------
# 1. REGRESIÓN LOGÍSTICA
# -------------------------------------------------------------------------------------
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
joblib.dump(logistic_model, "models/logistic_regression.pkl")

# -------------------------------------------------------------------------------------
# 2. SVM
# -------------------------------------------------------------------------------------
svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train_scaled, y_train)
joblib.dump(svm_model, "models/svm.pkl")

# -------------------------------------------------------------------------------------
# 3. RED NEURONAL
# -------------------------------------------------------------------------------------
neural_model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])
neural_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
neural_model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=0)
neural_model.save("models/neural_network.h5")

# -------------------------------------------------------------------------------------
# 4. FCM (Fuzzy C-Means)
# -------------------------------------------------------------------------------------
# --- Preparar datos para FCM (convertimos a la forma requerida por skfuzzy) ---
X_T = X_train_scaled.T  # (30, n_samples)

# --- Ajustar parámetros de FCM ---
n_clusters = 2  # Dos clusters (FGR y Normal)
m = 2.5  # Mayor difusividad
error = 0.001  # Mejor convergencia
maxiter = 3000  # Más iteraciones para ajuste preciso

# --- Entrenar FCM ---
centroids, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_T, c=n_clusters, m=m, error=error, maxiter=maxiter)

# --- Asignación de etiquetas por mayoría ---
cluster_assignments = np.argmax(u, axis=0)
cluster_labels = []
for k in range(n_clusters):
    indices = np.where(cluster_assignments == k)[0]
    if len(indices) == 0:
        cluster_labels.append(0)  # Cluster vacío → asignar clase 0
    else:
        labels_k = y_balanced[indices]
        majority_label = np.bincount(labels_k).argmax()
        cluster_labels.append(majority_label)

# --- Guardar modelo FCM ---
fcm_model = {
    "centroids": centroids,
    "cluster_labels": cluster_labels,
    "m": m,
    "error": error,
    "maxiter": maxiter
}
joblib.dump(fcm_model, "models/fcm.pkl")

print("🚀 Entrenamiento completado. Modelos guardados en la carpeta 'models'. ✅")
