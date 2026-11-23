# -*- coding: utf-8 -*-
"""Constantes de estilo y configuración visual global."""

import matplotlib.pyplot as plt

PALETTE = {
    "Autopista": "#1f77b4",
    "Urbana": "#ff7f0e",
    "Montaña": "#d62728",
}

ORDER = ["Autopista", "Urbana", "Montaña"]

VAR_MAP = {
    "Consumo de Combustible (Litros)": "Consumo_Litros",
    "Peso de Carga (Toneladas)": "Peso_Carga_Ton",
    "Velocidad Promedio (km/h)": "Velocidad_Promedio_kmh",
}


def configurar_matplotlib() -> None:
    """Aplica el estilo visual estándar a todas las gráficas."""
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 130,
    })
