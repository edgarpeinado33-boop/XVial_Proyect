"""
XVial – Controlador de Navegación e Interfaz Interactiva
Patrón de Comportamiento: Command (cada acción de botón es un Command)
Patrón Creacional: Factory Method (crea comandos según el botón presionado)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

from models.pavimento_model import PavimentoModel
from views.charts import ChartComposite


# ──────────────────────────────────────────────────────────────────────
# PATRÓN COMPORTAMIENTO: Command – interfaz base
# ──────────────────────────────────────────────────────────────────────
class NavigationCommand:
    def __init__(self, controller):
        self._ctrl = controller

    def execute(self, event):
        raise NotImplementedError


class NextCommand(NavigationCommand):
    def execute(self, event):
        self._ctrl.next_chart()


class PrevCommand(NavigationCommand):
    def execute(self, event):
        self._ctrl.prev_chart()


class GoToCommand(NavigationCommand):
    def __init__(self, controller, index):
        super().__init__(controller)
        self._index = index

    def execute(self, event):
        self._ctrl.go_to(self._index)


class ExportCommand(NavigationCommand):
    def execute(self, event):
        self._ctrl.export_all()


# ──────────────────────────────────────────────────────────────────────
# PATRÓN CREACIONAL: Factory Method – crea comandos
# ──────────────────────────────────────────────────────────────────────
class CommandFactory:
    @staticmethod
    def create(action: str, controller, index: int = 0) -> NavigationCommand:
        mapping = {
            "next":   lambda: NextCommand(controller),
            "prev":   lambda: PrevCommand(controller),
            "goto":   lambda: GoToCommand(controller, index),
            "export": lambda: ExportCommand(controller),
        }
        factory_fn = mapping.get(action)
        if factory_fn is None:
            raise ValueError(f"Acción desconocida: {action}")
        return factory_fn()


# ──────────────────────────────────────────────────────────────────────
# Controlador principal
# ──────────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
TEXT    = "#e6edf3"
ACCENT  = "#58a6ff"
BTN_ACT = "#1f6feb"
BTN_DEF = "#21262d"
BTN_TXT = "#e6edf3"


class XVialController:
    """
    Controlador MVC: gestiona la navegación entre gráficas,
    mantiene el estado de la vista activa y dispara los Commands.
    """

    def __init__(self):
        self._model  = PavimentoModel()          # Singleton
        self._charts = ChartComposite()          # Composite de vistas
        self._current = 4                        # Iniciar en Panel Predicción
        self._fig = None
        self._ax_main = None
        self._btn_objects: list[Button] = []
        self._tab_buttons: list[Button] = []

    # ── Construcción de la ventana ────────────────────────────────────
    def build_ui(self):
        plt.rcParams.update({
            "figure.facecolor":  DARK_BG,
            "text.color":        TEXT,
            "axes.facecolor":    CARD_BG,
            "axes.edgecolor":    "#30363d",
            "xtick.color":       TEXT,
            "ytick.color":       TEXT,
            "font.family":       "DejaVu Sans",
        })

        self._fig = plt.figure(figsize=(14, 8.5))
        self._fig.patch.set_facecolor(DARK_BG)
        self._fig.canvas.manager.set_window_title("XVial – Sistema de Predicción de Pavimento")

        # ── Encabezado ────────────────────────────────────────────────
        self._fig.text(0.5, 0.965, "XVial", ha="center", va="top",
                       fontsize=22, fontweight="bold", color=ACCENT,
                       fontfamily="DejaVu Sans")
        self._fig.text(0.5, 0.940, "Sistema de Predicción de Pavimento · Av. Blanco Galindo",
                       ha="center", va="top", fontsize=9, color="#8b949e")

        # ── Layout principal ─────────────────────────────────────────
        # Zona de gráfica principal
        self._ax_main = self._fig.add_axes([0.03, 0.13, 0.94, 0.78])
        self._ax_main.set_facecolor(CARD_BG)

        # ── Tabs de navegación (fila superior) ───────────────────────
        n = len(self._charts)
        tab_w = 0.105
        tab_gap = 0.007
        total_w = n * tab_w + (n - 1) * tab_gap
        start_x = (1.0 - total_w) / 2

        self._tab_buttons = []
        for i, name in enumerate(self._charts.names()):
            ax_tab = self._fig.add_axes([
                start_x + i * (tab_w + tab_gap),
                0.915, tab_w, 0.030,
            ])
            btn = Button(ax_tab, name,
                         color=BTN_DEF, hovercolor="#388bfd22")
            btn.label.set_fontsize(6.5)
            btn.label.set_color(BTN_TXT)
            cmd = CommandFactory.create("goto", self, index=i)
            btn.on_clicked(cmd.execute)
            self._tab_buttons.append(btn)

        # ── Botones Anterior / Siguiente ─────────────────────────────
        ax_prev = self._fig.add_axes([0.03,  0.015, 0.10, 0.038])
        ax_next = self._fig.add_axes([0.87,  0.015, 0.10, 0.038])
        ax_exp  = self._fig.add_axes([0.435, 0.015, 0.13, 0.038])

        self._btn_prev = Button(ax_prev, "◀  Anterior", color=BTN_DEF, hovercolor=BTN_ACT)
        self._btn_next = Button(ax_next, "Siguiente  ▶", color=BTN_DEF, hovercolor=BTN_ACT)
        self._btn_exp  = Button(ax_exp,  "⬇ Exportar PNG", color="#1a3a2a", hovercolor="#2ea043")

        for btn in [self._btn_prev, self._btn_next, self._btn_exp]:
            btn.label.set_color(BTN_TXT)
            btn.label.set_fontsize(8)

        cmd_prev = CommandFactory.create("prev",   self)
        cmd_next = CommandFactory.create("next",   self)
        cmd_exp  = CommandFactory.create("export", self)

        self._btn_prev.on_clicked(cmd_prev.execute)
        self._btn_next.on_clicked(cmd_next.execute)
        self._btn_exp.on_clicked(cmd_exp.execute)

        # ── Indicador de posición ─────────────────────────────────────
        self._pos_text = self._fig.text(
            0.5, 0.022, "", ha="center", va="bottom",
            fontsize=8, color="#8b949e"
        )

        self._render_current()
        return self._fig

    # ── Métodos de navegación ─────────────────────────────────────────
    def next_chart(self):
        self._current = (self._current + 1) % len(self._charts)
        self._render_current()

    def prev_chart(self):
        self._current = (self._current - 1) % len(self._charts)
        self._render_current()

    def go_to(self, index: int):
        self._current = index
        self._render_current()

    # ── Renderizado ───────────────────────────────────────────────────
    def _render_current(self):
        self._ax_main.clear()
        name, chart = self._charts[self._current]
        chart.render(self._ax_main, self._model)

        # Actualizar estilos de tabs
        for i, btn in enumerate(self._tab_buttons):
            active = i == self._current
            btn.ax.set_facecolor(BTN_ACT if active else BTN_DEF)
            btn.label.set_color("#ffffff" if active else BTN_TXT)
            btn.label.set_fontweight("bold" if active else "normal")

        # Indicador numérico
        self._pos_text.set_text(
            f"Gráfica {self._current + 1} de {len(self._charts)} · {name}"
        )
        self._fig.canvas.draw_idle()

    # ── Exportar todas las gráficas ───────────────────────────────────
    def export_all(self, event=None):
        fig_exp, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig_exp.patch.set_facecolor(DARK_BG)
        fig_exp.suptitle(
            "XVial – Dashboard Completo · Av. Blanco Galindo",
            color=ACCENT, fontsize=14, fontweight="bold", y=0.98,
        )
        axes_flat = axes.flatten()
        for i, (name, chart) in enumerate(self._charts):
            chart.render(axes_flat[i], self._model)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = "xvial_dashboard.png"
        fig_exp.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
        plt.close(fig_exp)
        self._pos_text.set_text(f"✅ Dashboard exportado → {path}")
        self._fig.canvas.draw_idle()
        print(f"[XVial] Dashboard exportado: {path}")

    def run(self):
        self.build_ui()
        plt.show()
