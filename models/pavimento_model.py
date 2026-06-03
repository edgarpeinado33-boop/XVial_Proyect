"""
XVial – Modelo de Datos y Machine Learning
Patrón Creacional: Singleton (instancia única del modelo)
Patrón Estructural: Facade (oculta la complejidad del pipeline ML)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────
# PATRÓN CREACIONAL: Singleton
# ──────────────────────────────────────────
class PavimentoModel:
    """
    Singleton que encapsula todos los datos y el modelo ML.
    Garantiza una única instancia durante toda la ejecución.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # ── Datos reales de aforo vehicular (30 min, Av. Blanco Galindo) ──
        self.aforo_real = {
            "Trufis / Camionetas": 196,
            "Micros":              58,
            "Autos particulares":  208,
            "Motos":               86,
            "Camiones (pesados)":  15,
        }
        self.total_vehiculos = sum(self.aforo_real.values())  # 563

        # ── Factores ESAL por tipo de vehículo ──
        self.esal_factors = {
            "Trufis / Camionetas": 0.5,
            "Micros":              2.0,
            "Autos particulares":  0.1,
            "Motos":               0.01,
            "Camiones (pesados)": 10.0,
        }

        # ── Dataset sintético enriquecido ──
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            "ESAL_equivalente",
            "antiguedad_pavimento",
            "lluvia_acumulada",
            "temperatura",
            "flujo_horario",
        ]
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.probabilities = None

        # Resultado de predicción para el aforo real
        self.prediccion_resultado = None
        self.prediccion_probabilidad = None

        self._build_dataset()
        self._train_model()
        self._predict_real_aforo()

    # ── PATRÓN ESTRUCTURAL: Facade ──────────────────────────────────────
    def _build_dataset(self):
        """Genera dataset sintético realista para entrenamiento."""
        np.random.seed(42)
        n = 500

        esal       = np.random.uniform(50, 3000, n)
        antiguedad = np.random.uniform(1, 25, n)
        lluvia     = np.random.uniform(0, 200, n)
        temp       = np.random.uniform(10, 35, n)
        flujo      = np.random.uniform(200, 2000, n)

        # Regla de negocio para etiquetado
        score = (
            (esal > 800).astype(int) * 3 +
            (antiguedad > 15).astype(int) * 2 +
            (lluvia > 100).astype(int) * 2 +
            (flujo > 1200).astype(int) * 2 +
            (temp > 28).astype(int)
        )
        necesita_reparacion = (score >= 4).astype(int)

        self.df = pd.DataFrame({
            "ESAL_equivalente":    esal,
            "antiguedad_pavimento": antiguedad,
            "lluvia_acumulada":    lluvia,
            "temperatura":         temp,
            "flujo_horario":       flujo,
            "necesita_reparacion": necesita_reparacion,
        })

    def _train_model(self):
        """Entrena el Random Forest Classifier."""
        X = self.df[self.feature_names].values
        y = self.df["necesita_reparacion"].values

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        self.X_test = X_test
        self.y_test = y_test

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)

    def _predict_real_aforo(self):
        """Realiza predicción con los datos reales del aforo de Av. Blanco Galindo."""
        esal_total = sum(
            self.aforo_real[veh] * self.esal_factors[veh]
            for veh in self.aforo_real
        )
        flujo_horario = self.total_vehiculos * 2  # extrapolar a 1 h

        muestra = np.array([[
            esal_total,   # ESAL equivalente
            12,           # antigüedad estimada pavimento (años)
            45,           # lluvia acumulada promedio (mm)
            22,           # temperatura media (°C)
            flujo_horario,
        ]])
        muestra_scaled = self.scaler.transform(muestra)
        self.prediccion_resultado   = int(self.model.predict(muestra_scaled)[0])
        proba = self.model.predict_proba(muestra_scaled)[0]
        self.prediccion_probabilidad = float(proba[self.prediccion_resultado])
        self.probabilities = proba

    # ── Accessors ───────────────────────────────────────────────────────
    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def get_feature_importances(self):
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def get_report(self):
        return classification_report(self.y_test, self.y_pred,
                                     target_names=["Sin reparación", "Necesita reparación"])

    def get_esal_total(self):
        return sum(
            self.aforo_real[v] * self.esal_factors[v] for v in self.aforo_real
        )
