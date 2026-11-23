# -*- coding: utf-8 -*-
"""Pestaña 4: Simulador Interactivo de Consumo y Costos Operativos."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.config import ORDER, PALETTE
from src.stats_engine import predecir_con_ic


def render_simulador(modelo_base: dict, modelo_ext: dict) -> None:
    st.subheader("4. Simulador Interactivo de Consumo y Costos Operativos")
    st.markdown(
        "Modifica el peso de la carga, la velocidad promedio, el tipo de ruta y el precio del diésel para "
        "explorar distintos escenarios operativos en tiempo real."
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### Parámetros del Viaje")
        sim_peso = st.slider("Peso de carga (Toneladas)", min_value=4.0, max_value=30.0, value=15.0, step=0.5)
        sim_vel = st.slider("Velocidad promedio (km/h)", min_value=20.0, max_value=110.0, value=80.0, step=1.0)
        sim_ruta = st.selectbox("Tipo de ruta (para estimación ajustada)", options=ORDER, index=0)

    with p_col2:
        st.markdown("#### Parámetros Financieros")
        precio_diesel = st.number_input("Precio del diésel (MXN / Litro)", min_value=10.0, max_value=50.0, value=25.50, step=0.50)
        viajes_mes = st.slider("Viajes mensuales estimados", min_value=10, max_value=500, value=100, step=10)

    pred_base = predecir_con_ic(modelo_base, [sim_peso, sim_vel])
    d_urb = 1.0 if sim_ruta == "Urbana" else 0.0
    d_mon = 1.0 if sim_ruta == "Montaña" else 0.0
    pred_ruta = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, d_urb, d_mon])

    pred_auto = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, 0.0, 0.0])["pred"]
    pred_urb = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, 1.0, 0.0])["pred"]
    pred_mon = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, 0.0, 1.0])["pred"]

    st.divider()
    st.markdown("#### Resultados del Escenario Simulado")
    res1, res2, res3 = st.columns(3)
    res1.metric(
        "Consumo (Modelo Base)",
        f"{pred_base['pred']:.2f} L",
        f"IC 95%: [{pred_base['ci_low']:.1f}, {pred_base['ci_high']:.1f}] L",
        delta_color="off",
    )
    res2.metric(
        f"Consumo ({sim_ruta})",
        f"{pred_ruta['pred']:.2f} L",
        f"Costo/viaje: ${pred_ruta['pred'] * precio_diesel:,.0f} MXN",
        delta_color="off",
    )
    ahorro_vs_mon = max(0.0, pred_mon - pred_auto) * precio_diesel * viajes_mes
    res3.metric(
        "Ahorro Mensual (Autopista vs. Montaña)",
        f"${ahorro_vs_mon:,.0f} MXN",
        f"{pred_mon - pred_auto:.1f} L menos por viaje",
    )

    _graficar_sensibilidad(
        modelo_base, modelo_ext, sim_peso, sim_vel, sim_ruta,
        d_urb, d_mon, pred_base["pred"], pred_ruta["pred"],
        [pred_auto, pred_urb, pred_mon],
    )


def _graficar_sensibilidad(
    modelo_base: dict,
    modelo_ext: dict,
    sim_peso: float,
    sim_vel: float,
    sim_ruta: str,
    d_urb: float,
    d_mon: float,
    y_base: float,
    y_ruta: float,
    rutas_vals: list[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    pesos_grid = np.linspace(4, 30, 60)
    y_grid_base = modelo_base["beta"][0] + modelo_base["beta"][1] * pesos_grid + modelo_base["beta"][2] * sim_vel
    y_grid_ruta = (
        modelo_ext["beta"][0]
        + modelo_ext["beta"][1] * pesos_grid
        + modelo_ext["beta"][2] * sim_vel
        + modelo_ext["beta"][3] * d_urb
        + modelo_ext["beta"][4] * d_mon
    )
    axes[0].plot(pesos_grid, y_grid_base, color="#333333", linestyle="--", linewidth=1.8, label="Modelo Base")
    axes[0].plot(pesos_grid, y_grid_ruta, color=PALETTE[sim_ruta], linewidth=2.2, label=f"Modelo ({sim_ruta})")
    axes[0].scatter([sim_peso], [y_base], color="#333333", s=65, zorder=5)
    axes[0].scatter([sim_peso], [y_ruta], color=PALETTE[sim_ruta], s=75, zorder=5, edgecolor="black")
    axes[0].set_xlabel("Peso de Carga (Toneladas)")
    axes[0].set_ylabel("Consumo Estimado (Litros)")
    axes[0].set_title(f"Sensibilidad al Peso (a {sim_vel:.0f} km/h)")
    axes[0].legend(fontsize=8.5)

    bars = axes[1].bar(ORDER, rutas_vals, color=[PALETTE[r] for r in ORDER], alpha=0.85, edgecolor="#222")
    for bar, v in zip(bars, rutas_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f} L",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Consumo Estimado (Litros)")
    axes[1].set_ylim(0, max(rutas_vals) * 1.18)
    axes[1].set_title(f"Consumo por Ruta ({sim_peso:.1f} t, {sim_vel:.0f} km/h)")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
