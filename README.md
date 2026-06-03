# 🛣️ XVial – Sistema de Predicción de Pavimento

**Av. Blanco Galindo · Cochabamba, Bolivia**

Sistema de Machine Learning con interfaz web responsiva, gráficos 3D interactivos
y cumplimiento de normas **ISO/IEC 25010** e **ISO/IEC 25020**.

---

## 🚀 Uso — Un solo comando

```bash
python main.py
```

Al ejecutar `main.py`:

1. **Entrena** el modelo Random Forest con los datos reales de aforo
2. **Imprime** el reporte del clasificador en consola
3. **Genera** `data.json` con todos los resultados del modelo
4. **Levanta** un servidor HTTP local (puerto 8765)
5. **Abre** automáticamente el dashboard en el navegador
6. El dashboard **carga los datos reales** del modelo Python via `data.json`

El servidor se mantiene activo hasta que presiones **Ctrl+C**.

---

## ⚙️ Instalación

```bash
# 1. Entorno virtual (recomendado)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Dependencias
pip install -r requirements.txt

# 3. Ejecutar
python main.py
```

---

## 🗂️ Estructura del proyecto

```
xvial/
├── main.py                         ← Punto de entrada (servidor + ML)
├── index.html                      ← Dashboard web responsiva (3D, ISO)
├── models/
│   └── pavimento_model.py          ← Modelo ML (Singleton + Facade)
├── views/
│   └── charts.py                   ← Vistas matplotlib (opcional)
├── controllers/
│   └── navigation_controller.py   ← Controlador (Command + Factory)
└── requirements.txt
```

---

## ⚙️ Opciones en main.py

| Variable | Default | Descripción |
|---|---|---|
| `SERVER_PORT` | `8765` | Puerto del servidor local |
| `OPEN_BROWSER` | `True` | Abre el navegador automáticamente |
| `RUN_MATPLOTLIB` | `False` | Abre también la ventana matplotlib |

Para activar la ventana matplotlib además del dashboard web:
```python
RUN_MATPLOTLIB = True
```

---

## 🔄 Flujo de datos Python → Web

```
main.py
  │
  ├─ PavimentoModel()          ← Entrena Random Forest
  │    ├─ aforo_real            (datos reales Av. Blanco Galindo)
  │    ├─ prediccion_resultado  (0 = sin rep. / 1 = necesita rep.)
  │    ├─ prediccion_probabilidad
  │    ├─ confusion_matrix
  │    └─ feature_importances
  │
  ├─ build_data_json()         ← Serializa resultados a dict
  ├─ prepare_web_dir()         ← Crea directorio temporal con data.json
  ├─ start_server()            ← HTTP en localhost:8765
  └─ webbrowser.open()         ← Abre el navegador
           │
           ▼
     index.html
       └─ fetch('data.json')   ← Carga datos reales del modelo Python
            └─ Gráficos 3D actualizados con valores reales
```

---

## 📋 Cumplimiento ISO/IEC 25010

| Característica | Implementación |
|---|---|
| Funcionalidad | 9 pestañas, modelo ML real, datos de aforo reales |
| Fiabilidad | Fallback a valores precalculados si data.json no disponible |
| Usabilidad | WCAG AA, ARIA, navegación teclado/swipe táctil |
| Eficiencia | Lazy render, servidor en hilo daemon |
| Mantenibilidad | MVC, variables CSS, funciones puras |
| Portabilidad | Responsive mobile/tablet/desktop |
| Seguridad | Sin eval(), servidor local solo, sin acceso externo |
| Compatibilidad | Plotly CDN, HTML5/CSS3, Python stdlib |

## 📏 Métricas ISO/IEC 25020

| Métrica | Valor |
|---|---|
| Precisión del clasificador | ~84.8% |
| Recall (sensibilidad) | ~84.9% |
| F1-Score | ~84.7% |
| Aforo vehicular | 563 vehículos (30 min) |
| ESAL total | calculado en tiempo real |
| Confianza de predicción | calculado en tiempo real |

---

## 📊 Datos reales (30 min · Av. Blanco Galindo)

| Tipo de vehículo    | Cantidad |
|---------------------|----------|
| Trufis / Camionetas | 196      |
| Micros              | 58       |
| Autos particulares  | 208      |
| Motos               | 86       |
| Camiones (pesados)  | 15       |
| **TOTAL**           | **563**  |

---

*Desarrollado con datos reales de la Av. Blanco Galindo – Cochabamba, Bolivia.*
