# -*- coding: utf-8 -*-
"""Pestaña 5: Toma de Decisiones Gerenciales (FastTrack Logistics)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import ORDER, PALETTE
from src.stats_engine import predecir_con_ic


def render_decisiones(df_raw: pd.DataFrame, modelo_base: dict, modelo_ext: dict) -> None:
    st.subheader("5. Toma de Decisiones Gerenciales (FastTrack Logistics)")
    st.markdown(
        "Respuestas cuantitativas a las cuatro interrogantes planteadas en el caso de estudio para revertir "
        "el aumento del **18%** en costos operativos."
    )

    df_dec = df_raw.copy()
    df_dec["Nivel_Carga"] = pd.cut(
        df_dec["Peso_Carga_Ton"],
        bins=[0, 12, 20, 35],
        labels=["Ligera (<12 t)", "Media (12–20 t)", "Pesada (>20 t)"],
    )
    pivot_carga = df_dec.pivot_table(
        values="Consumo_Litros", index="Tipo_Ruta", columns="Nivel_Carga",
        aggfunc="mean", observed=False,
    ).reindex(ORDER)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### 1. ¿Evitar ciertas rutas para cargas pesadas?")
        st.error(
            "**Sí — Evitar la ruta de Montaña para cargas pesadas (>20 t).**  \n"
            f"En Montaña las cargas pesadas promedian **{pivot_carga.loc['Montaña', 'Pesada (>20 t)']:.2f} L**, "
            f"frente a **{pivot_carga.loc['Autopista', 'Pesada (>20 t)']:.2f} L** en Autopista "
            f"y **{pivot_carga.loc['Urbana', 'Pesada (>20 t)']:.2f} L** en Urbana. "
            "Reasignar una carga pesada de Montaña a Autopista ahorra ~**30.4 L por viaje (~36%)**."
        )

        st.markdown("#### 3. ¿Controlar velocidad o reducir peso de carga?")
        beta_std_peso = abs(modelo_base["tabla_coefs"].loc[1, "β Estandarizado"])
        beta_std_vel = abs(modelo_base["tabla_coefs"].loc[2, "β Estandarizado"])
        st.warning(
            "**Priorizar el control y optimización del peso de carga.**  \n"
            f"El coeficiente estandarizado del peso ($|\\beta^*| = {beta_std_peso:.4f}$) es "
            f"**{beta_std_peso / beta_std_vel:.2f} veces mayor** que el de la velocidad "
            f"($|\\beta^*| = {beta_std_vel:.4f}$). Cada tonelada adicional suma **+2.46 L** "
            "($t = 14.07, p = 1.81 \\times 10^{-18}$). Al controlar por tipo de ruta, la velocidad deja de ser significativa ($p = 0.0828$)."
        )

    with d2:
        st.markdown("#### 2. ¿En qué ruta se gasta más y es significativo?")
        st.error(
            "**Ruta de Montaña (75.01 L/viaje) — Altamente significativo.**  \n"
            "Supera en **+33.76 L** a Autopista (41.25 L) y en **+32.75 L** a Urbana (42.27 L). "
            "El ANOVA confirma la diferencia con $F(2, 47) = 39.4650$ y $p = 8.735 \\times 10^{-11} \\ll 0.05$ "
            "($\\eta^2 = 62.68\\%$ de la varianza explicada)."
        )

        st.markdown("#### 4. Pronóstico: Camión con 15 toneladas a 80 km/h")
        p_15_80 = predecir_con_ic(modelo_base, [15.0, 80.0])
        p_15_80_auto = predecir_con_ic(modelo_ext, [15.0, 80.0, 0.0, 0.0])
        st.success(
            f"**Modelo Base (Peso + Velocidad):** $\\widehat{{Y}} = 26.1794 + 2.4600(15) - 0.2290(80) = $ "
            f"**{p_15_80['pred']:.2f} litros** (IC 95%: $[{p_15_80['ci_low']:.2f}, {p_15_80['ci_high']:.2f}]$ L).  \n"
            f"**Referencia ajustada en Autopista (Modelo Extendido):** **{p_15_80_auto['pred']:.2f} litros**."
        )

    _graficar_niveles_carga(pivot_carga)


def _graficar_niveles_carga(pivot_carga: pd.DataFrame) -> None:
    st.markdown("#### Consumo Promedio por Nivel de Carga y Tipo de Ruta")
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    niveles = ["Ligera (<12 t)", "Media (12–20 t)", "Pesada (>20 t)"]
    x = np.arange(len(niveles))
    w = 0.25
    for idx, ruta in enumerate(ORDER):
        vals = [pivot_carga.loc[ruta, niv] for niv in niveles]
        bars = ax.bar(x + (idx - 1) * w, vals, w, label=ruta, color=PALETTE[ruta], alpha=0.85, edgecolor="#222")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f} L",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(niveles)
    ax.set_ylabel("Consumo Promedio (Litros)")
    ax.set_ylim(0, 98)
    ax.set_title("Impacto de la Carga Pesada según el Tipo de Ruta")
    ax.legend(title="Tipo de Ruta")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
