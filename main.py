"""
XVial – Punto de entrada principal
Sistema de Predicción de Pavimento · Av. Blanco Galindo, Cochabamba

Arquitectura: MVC
Patrones:
  • Creacional  → Singleton (PavimentoModel), Factory Method (CommandFactory)
  • Estructural → Facade (pipeline ML), Composite (ChartComposite)
  • Comportamiento → Observer (ChartObserver), Command (NavigationCommand)

ISO/IEC 25010 – Portabilidad / Funcionalidad:
  Al ejecutar main.py se inicia un servidor HTTP local y se abre
  automáticamente el dashboard web (index.html) en el navegador
  predeterminado del sistema, pasándole los datos reales calculados
  por el modelo Python vía un archivo JSON temporal (data.json).

ISO/IEC 25020 – Trazabilidad de datos:
  Los valores mostrados en la interfaz web provienen directamente
  del modelo scikit-learn entrenado en este proceso Python,
  garantizando consistencia entre la lógica ML y la visualización.
"""

import sys
import os
import json
import time
import threading
import webbrowser
import http.server
import socketserver
import shutil
import tempfile

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.pavimento_model import PavimentoModel

# ── Configuración del servidor ────────────────────────────────────────
SERVER_PORT  = 8765          # Puerto local; se autoincrementa si está ocupado
OPEN_BROWSER = True          # False para deshabilitar apertura automática
RUN_MATPLOTLIB = False       # True para abrir también la ventana matplotlib


# ══════════════════════════════════════════════════════════════════════
def build_data_json(model: PavimentoModel) -> dict:
    """
    Serializa los resultados del modelo Python a un dict JSON.
    El index.html leerá /data.json en el servidor local para
    reemplazar sus valores precalculados con los datos reales.

    ISO 25020 – Medición: todos los valores son calculados por el
    clasificador Random Forest entrenado en este proceso.
    """
    importancias = model.get_feature_importances()
    cm           = model.get_confusion_matrix().tolist()
    probs        = model.probabilities.tolist()

    return {
        # ── Aforo real ──────────────────────────────────────────────
        "aforo_real": model.aforo_real,
        "total_vehiculos": model.total_vehiculos,
        "esal_total": round(model.get_esal_total(), 2),

        # ── Resultados del modelo ───────────────────────────────────
        "prediccion_resultado":     model.prediccion_resultado,
        "prediccion_probabilidad":  round(model.prediccion_probabilidad, 4),
        "probabilities":            [round(p, 4) for p in probs],

        # ── Métricas del clasificador ───────────────────────────────
        "confusion_matrix":   cm,
        "feature_importances": {k: round(v, 4) for k, v in importancias.items()},

        # ── Reporte de texto (para consola y panel ISO) ─────────────
        "classification_report": model.get_report(),

        # ── Flujo horario para proyección ───────────────────────────
        "flujo_horario": model.total_vehiculos * 2,

        # ── Metadatos ───────────────────────────────────────────────
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_params": {
            "algorithm":    "RandomForestClassifier",
            "n_estimators": 150,
            "max_depth":    8,
            "random_state": 42,
            "class_weight": "balanced",
        },
    }


# ══════════════════════════════════════════════════════════════════════
def prepare_web_dir(data: dict) -> str:
    """
    Crea un directorio temporal con los archivos web y el data.json
    generado por el modelo. Devuelve la ruta del directorio.

    ISO 25010 – Mantenibilidad: separación clara entre el servidor
    temporal y los archivos fuente del proyecto.
    """
    # Directorio temporal para servir
    web_dir = tempfile.mkdtemp(prefix="xvial_serve_")

    # Copiar index.html al directorio temporal
    src_html = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(src_html):
        raise FileNotFoundError(
            f"No se encontró index.html en {BASE_DIR}\n"
            "Asegúrate de que index.html está en la misma carpeta que main.py."
        )
    shutil.copy2(src_html, os.path.join(web_dir, "index.html"))

    # Escribir data.json con los datos reales del modelo
    data_path = os.path.join(web_dir, "data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return web_dir


# ══════════════════════════════════════════════════════════════════════
class SilentHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """
    Handler HTTP que silenció los logs de acceso para no ensuciar
    la consola durante la sesión interactiva.

    ISO 25010 – Usabilidad: salida de consola limpia y legible.
    """
    def log_message(self, format, *args):
        pass   # suprimir logs de acceso

    def log_error(self, format, *args):
        pass   # suprimir errores menores (favicon 404, etc.)


def find_free_port(start: int) -> int:
    """Busca el primer puerto libre a partir de `start`."""
    import socket
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No se encontró un puerto libre en el rango buscado.")


def start_server(web_dir: str, port: int) -> socketserver.TCPServer:
    """
    Levanta el servidor HTTP en un hilo daemon para que no bloquee
    el proceso principal ni quede huérfano al cerrar.

    ISO 25010 – Fiabilidad: el servidor se limpia automáticamente
    al terminar el proceso Python.
    """
    os.chdir(web_dir)

    httpd = socketserver.TCPServer(("", port), SilentHTTPHandler)
    httpd.allow_reuse_address = True

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# ══════════════════════════════════════════════════════════════════════
def print_banner(model: PavimentoModel, port: int):
    """Imprime el reporte del modelo en consola."""
    W = 62
    sep = "═" * W

    resultado_txt = (
        "✅  NO necesita reparación"
        if model.prediccion_resultado == 0
        else "⚠️   SÍ necesita reparación"
    )

    print(f"\n{sep}")
    print(f"  XVial – Sistema de Predicción de Pavimento")
    print(f"  Av. Blanco Galindo · Cochabamba, Bolivia")
    print(sep)

    print(f"\n  Vehículos totales (30 min) : {model.total_vehiculos}")
    print(f"  ESAL equivalente total     : {model.get_esal_total():.1f}")

    print(f"\n  ─── PREDICCIÓN ───────────────────────────────────────")
    print(f"  {resultado_txt}")
    print(f"  Probabilidad de confianza  : {model.prediccion_probabilidad * 100:.1f}%")

    print(f"\n  ─── REPORTE DEL CLASIFICADOR ─────────────────────────")
    for line in model.get_report().splitlines():
        print(f"  {line}")

    print(f"\n{sep}")
    print(f"  🌐 Dashboard web disponible en:")
    print(f"     http://localhost:{port}/index.html")
    print(f"\n  El navegador debería abrirse automáticamente.")
    print(f"  Presiona Ctrl+C para detener el servidor.")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════
def main():
    # ── 1. Entrenar modelo ────────────────────────────────────────────
    print("\n  Iniciando modelo de Machine Learning...")
    model = PavimentoModel()

    # ── 2. Serializar datos reales a JSON ─────────────────────────────
    data = build_data_json(model)

    # ── 3. Preparar directorio web con data.json ──────────────────────
    web_dir = prepare_web_dir(data)

    # ── 4. Buscar puerto libre y levantar servidor ────────────────────
    port = find_free_port(SERVER_PORT)
    httpd = start_server(web_dir, port)

    # ── 5. Imprimir resultados en consola ─────────────────────────────
    print_banner(model, port)

    # ── 6. Abrir navegador ────────────────────────────────────────────
    url = f"http://localhost:{port}/index.html"
    if OPEN_BROWSER:
        # Pequeña pausa para que el servidor esté listo
        time.sleep(0.4)
        webbrowser.open(url)

    # ── 7. (Opcional) Abrir también la ventana matplotlib ─────────────
    if RUN_MATPLOTLIB:
        try:
            from controllers.navigation_controller import XVialController
            ctrl = XVialController()
            ctrl.run()          # bloquea hasta cerrar la ventana
        except Exception as e:
            print(f"  [matplotlib] No se pudo abrir la interfaz gráfica: {e}")

    # ── 8. Mantener el servidor vivo ──────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Deteniendo servidor XVial...")
        httpd.shutdown()
        shutil.rmtree(web_dir, ignore_errors=True)
        print("  ¡Hasta luego!\n")


if __name__ == "__main__":
    main()
