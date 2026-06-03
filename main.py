"""
XVial – Punto de entrada principal (Dual: Vercel Serverless + Servidor Local)
Sistema de Predicción de Pavimento · Av. Blanco Galindo, Cochabamba

Arquitectura: MVC Intacta
Patrones:
  • Creacional  → Singleton (PavimentoModel), Factory Method (CommandFactory)
  • Estructural → Facade (pipeline ML), Composite (ChartComposite)
  • Comportamiento → Observer (ChartObserver), Command (NavigationCommand)

ISO/IEC 25010 – Portabilidad / Funcionalidad:
  Soporta ejecución híbrida: actúa como Serverless Function nativa en Vercel
  y como servidor HTTP autoinvocable de forma local.
"""

import sys
import os
import json
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
import socketserver
import shutil
import tempfile

# ── Path setup (Mantiene la carga modular de tu arquitectura) ────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.pavimento_model import PavimentoModel

# ── Configuración para el entorno de desarrollo local ────────────────
SERVER_PORT  = 8765          # Puerto local; se autoincrementa si está ocupado
OPEN_BROWSER = True          # False para deshabilitar apertura automática
RUN_MATPLOTLIB = False       # True para abrir también la ventana matplotlib


# ══════════════════════════════════════════════════════════════════════
def build_data_json(model: PavimentoModel) -> dict:
    """
    Serializa los resultados del modelo Python a un dict JSON.
    El index.html leerá /data.json para renderizar los gráficos 3D.
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
class handler(BaseHTTPRequestHandler):
    """
    MANEJADOR NATIVO PARA VERCEL SERVERLESS
    Mapea de forma directa las peticiones GET desde el entorno asíncrono en la nube.
    """
    def do_GET(self):
        if self.path == '/data.json' or self.path == '/main.py':
            model = PavimentoModel()
            data = build_data_json(model)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Ruta no encontrada")


# ══════════════════════════════════════════════════════════════════════
# LÓGICA DE TRANSPORTE Y OPERACIONES PARA EL ENTORNO LOCAL
# ══════════════════════════════════════════════════════════════════════

def prepare_web_dir(data: dict) -> str:
    """Crea un directorio temporal local con los archivos web y el data.json."""
    web_dir = tempfile.mkdtemp(prefix="xvial_serve_")
    src_html = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(src_html):
        raise FileNotFoundError(f"No se encontró index.html en {BASE_DIR}")
    
    shutil.copy2(src_html, os.path.join(web_dir, "index.html"))

    data_path = os.path.join(web_dir, "data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return web_dir


class SilentHTTPHandler(SimpleHTTPRequestHandler):
    """Manejador estático local con supresión de logs recurrentes."""
    def log_message(self, format, *args): pass
    def log_error(self, format, *args): pass


def find_free_port(start: int) -> int:
    import socket
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No se encontró un puerto libre.")


def start_local_server(web_dir: str, port: int) -> socketserver.TCPServer:
    os.chdir(web_dir)
    httpd = socketserver.TCPServer(("", port), SilentHTTPHandler)
    httpd.allow_reuse_address = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def print_banner(model: PavimentoModel, port: int):
    W = 62
    sep = "═" * W
    resultado_txt = "✅  NO necesita reparación" if model.prediccion_resultado == 0 else "⚠️   SÍ necesita reparación"

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
    print(f"  🌐 Dashboard web disponible localmente en:")
    print(f"     http://localhost:{port}/index.html")
    print(f"\n  El navegador debería abrirse automáticamente.")
    print(f"  Presiona Ctrl+C para detener el servidor.")
    print(f"{sep}\n")


# ── Entrada en modo Local (Ignorado automáticamente por Vercel) ───────
if __name__ == "__main__":
    print("\n  [Modo Local] Iniciando modelo de Machine Learning...")
    local_model = PavimentoModel()
    local_data = build_data_json(local_model)
    
    local_web_dir = prepare_web_dir(local_data)
    local_port = find_free_port(SERVER_PORT)
    local_httpd = start_local_server(local_web_dir, local_port)

    print_banner(local_model, local_port)

    if OPEN_BROWSER:
        time.sleep(0.4)
        webbrowser.open(f"http://localhost:{local_port}/index.html")

    if RUN_MATPLOTLIB:
        try:
            from controllers.navigation_controller import XVialController
            ctrl = XVialController()
            ctrl.run()
        except Exception as e:
            print(f"  [matplotlib] No se pudo abrir la interfaz gráfica local: {e}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Deteniendo servidor local XVial...")
        local_httpd.shutdown()
        shutil.rmtree(local_web_dir, ignore_errors=True)
        print("  ¡Hasta luego!\n")