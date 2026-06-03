"""
XVial – Vistas (generadores de gráficas matplotlib)
Patrón de Comportamiento: Observer (cada vista "observa" el modelo y se actualiza)
Patrón Estructural: Composite (todas las vistas componen el dashboard)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────────────────────────────
# PATRÓN COMPORTAMIENTO: Observer – interfaz base
# ──────────────────────────────────────────────────────────────────────
class ChartObserver(ABC):
    """Interfaz Observer: cada gráfica observa el modelo."""

    @abstractmethod
    def render(self, ax, model):
        """Dibuja la gráfica en el Axes proporcionado."""


# ──────────────────────────────────────────────────────────────────────
# Implementaciones concretas
# ──────────────────────────────────────────────────────────────────────
PALETTE = ["#1a73e8", "#ea4335", "#34a853", "#fbbc04", "#9c27b0"]
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
TEXT     = "#e6edf3"
ACCENT   = "#58a6ff"


def _style_ax(ax, title):
    ax.set_facecolor(CARD_BG)
    ax.title.set_color(ACCENT)
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("bold")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


class AforoBarChart(ChartObserver):
    """Gráfica 1 – Aforo real por tipo de vehículo."""

    def render(self, ax, model):
        labels = list(model.aforo_real.keys())
        values = list(model.aforo_real.values())
        bars = ax.bar(labels, values, color=PALETTE, edgecolor="#30363d", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(val), ha="center", va="bottom", color=TEXT, fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Cantidad")
        _style_ax(ax, f"📊 Aforo Real – Av. Blanco Galindo (total: {model.total_vehiculos})")


class ComposicionPieChart(ChartObserver):
    """Gráfica 2 – Composición porcentual del tráfico."""

    def render(self, ax, model):
        labels = list(model.aforo_real.keys())
        values = list(model.aforo_real.values())
        wedges, texts, autotexts = ax.pie(
            values, labels=None, autopct="%1.1f%%",
            colors=PALETTE, startangle=140,
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5},
            textprops={"color": TEXT, "fontsize": 7},
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color(DARK_BG)
        ax.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.25),
                  fontsize=6, ncol=2, labelcolor=TEXT,
                  facecolor=CARD_BG, edgecolor="#30363d")
        _style_ax(ax, "🥧 Composición Vehicular (%)")


class ImportanciaChart(ChartObserver):
    """Gráfica 3 – Importancia de variables del Random Forest."""

    def render(self, ax, model):
        importancias = model.get_feature_importances()
        nombres = [n.replace("_", "\n") for n in importancias]
        valores = list(importancias.values())
        sorted_pairs = sorted(zip(valores, nombres), reverse=True)
        valores, nombres = zip(*sorted_pairs)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(nombres))]
        bars = ax.barh(nombres, valores, color=colors, edgecolor="#30363d")
        for bar, val in zip(bars, valores):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", color=TEXT, fontsize=7)
        ax.set_xlabel("Importancia")
        ax.invert_yaxis()
        _style_ax(ax, "🧠 Importancia de Variables")


class MatrizConfusionChart(ChartObserver):
    """Gráfica 4 – Matriz de confusión."""

    def render(self, ax, model):
        cm = model.get_confusion_matrix()
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else TEXT,
                        fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Sin rep.", "Necesita rep."], color=TEXT, fontsize=8)
        ax.set_yticklabels(["Sin rep.", "Necesita rep."], color=TEXT, fontsize=8)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        _style_ax(ax, "🔢 Matriz de Confusión")


class PanelPrediccionChart(ChartObserver):
    """Gráfica 5 – Panel principal de predicción (verde / rojo)."""

    def render(self, ax, model):
        necesita = model.prediccion_resultado == 1
        color_bg   = "#1a3a2a" if not necesita else "#3a1a1a"
        color_bord = "#34a853" if not necesita else "#ea4335"
        texto_res  = "✅ SIN REPARACIÓN" if not necesita else "⚠️ NECESITA REPARACIÓN"
        color_text = "#34a853" if not necesita else "#ea4335"
        prob_pct   = model.prediccion_probabilidad * 100

        ax.set_facecolor(color_bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(color_bord)
            spine.set_linewidth(3)

        ax.text(0.5, 0.72, "PREDICCIÓN PARA", ha="center", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=9)
        ax.text(0.5, 0.60, "Av. Blanco Galindo", ha="center", va="center",
                transform=ax.transAxes, color=ACCENT, fontsize=11, fontweight="bold")
        ax.text(0.5, 0.42, texto_res, ha="center", va="center",
                transform=ax.transAxes, color=color_text, fontsize=15, fontweight="bold")
        ax.text(0.5, 0.25, f"Confianza: {prob_pct:.1f}%", ha="center", va="center",
                transform=ax.transAxes, color=TEXT, fontsize=10)
        esal = model.get_esal_total()
        ax.text(0.5, 0.10, f"ESAL total (30 min): {esal:.0f}", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e", fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("🚦 Panel de Predicción Principal", color=ACCENT,
                     fontsize=10, fontweight="bold")


class DistribucionProbChart(ChartObserver):
    """Gráfica 6 – Distribución de probabilidades."""

    def render(self, ax, model):
        labels = ["Sin reparación", "Necesita reparación"]
        values = [model.probabilities[0] * 100, model.probabilities[1] * 100]
        colors = ["#34a853", "#ea4335"]
        bars = ax.bar(labels, values, color=colors, edgecolor="#30363d", width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=10,
                    fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Probabilidad (%)")
        _style_ax(ax, "📉 Distribución de Probabilidades")


class AntiguedadCargaChart(ChartObserver):
    """Gráfica 7 – Antigüedad vs carga de tráfico (scatter)."""

    def render(self, ax, model):
        df = model.df
        colors_map = df["necesita_reparacion"].map({0: "#34a853", 1: "#ea4335"})
        ax.scatter(df["antiguedad_pavimento"], df["ESAL_equivalente"],
                   c=colors_map, alpha=0.5, s=15, edgecolors="none")
        patch_ok  = mpatches.Patch(color="#34a853", label="Sin reparación")
        patch_rep = mpatches.Patch(color="#ea4335", label="Necesita rep.")
        ax.legend(handles=[patch_ok, patch_rep], fontsize=7,
                  facecolor=CARD_BG, edgecolor="#30363d", labelcolor=TEXT)
        ax.set_xlabel("Antigüedad (años)")
        ax.set_ylabel("ESAL equivalente")
        _style_ax(ax, "📐 Antigüedad vs Carga de Tráfico")


class ProyeccionHorariaChart(ChartObserver):
    """Gráfica 8 – Proyección horaria del flujo vehicular."""

    def render(self, ax, model):
        horas = np.arange(0, 24)
        # Curva de demanda vehicular típica
        base = model.total_vehiculos * 2
        flujo = (
            base * (
                0.3 + 0.7 * np.exp(-((horas - 7.5)**2) / 4) +
                0.5 * np.exp(-((horas - 17.5)**2) / 5)
            )
        )
        ax.plot(horas, flujo, color=ACCENT, linewidth=2)
        ax.fill_between(horas, flujo, alpha=0.2, color=ACCENT)
        ax.axhline(base * 0.8, color="#ea4335", linestyle="--",
                   linewidth=1, label="Umbral crítico")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Veh / hora (proyección)")
        ax.legend(fontsize=7, facecolor=CARD_BG, edgecolor="#30363d", labelcolor=TEXT)
        _style_ax(ax, "⏱️ Proyección Horaria del Flujo")


# ──────────────────────────────────────────────────────────────────────
# PATRÓN ESTRUCTURAL: Composite – colección de vistas
# ──────────────────────────────────────────────────────────────────────
class ChartComposite:
    """Agrupa todas las vistas y las expone como una lista ordenada."""

    def __init__(self):
        self._charts: list[tuple[str, ChartObserver]] = [
            ("Aforo Real",          AforoBarChart()),
            ("Composición (%)",     ComposicionPieChart()),
            ("Variables ML",        ImportanciaChart()),
            ("Matriz Confusión",    MatrizConfusionChart()),
            ("Panel Predicción",    PanelPrediccionChart()),
            ("Probabilidades",      DistribucionProbChart()),
            ("Antigüedad vs Carga", AntiguedadCargaChart()),
            ("Proyección Horaria",  ProyeccionHorariaChart()),
        ]

    def __len__(self):
        return len(self._charts)

    def __getitem__(self, index):
        return self._charts[index]

    def names(self):
        return [name for name, _ in self._charts]
