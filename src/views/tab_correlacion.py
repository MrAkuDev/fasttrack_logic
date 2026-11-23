# -*- coding: utf-8 -*-
"""Pestaña 2: Análisis de Correlación de Pearson y Dispersión."""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st

from src.config import PALETTE


def render_correlacion(df: pd.DataFrame, rutas_sel: list[str]) -> None:
    st.subheader("2. Matriz de Correlación de Pearson y Dispersión")
    num_cols = ["Peso_Carga_Ton", "Velocidad_Promedio_kmh", "Consumo_Litros"]
    corr_mat = df[num_cols].corr(method="pearson")

    r_peso, p_peso = stats.pearsonr(df["Peso_Carga_Ton"], df["Consumo_Litros"])
    r_vel, p_vel = stats.pearsonr(df["Velocidad_Promedio_kmh"], df["Consumo_Litros"])
    r_pv, p_pv = stats.pearsonr(df["Peso_Carga_Ton"], df["Velocidad_Promedio_kmh"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Peso vs. Consumo (r)", f"{r_peso:.4f}", f"p = {p_peso:.2e}", delta_color="off")
    m2.metric("Velocidad vs. Consumo (r)", f"{r_vel:.4f}", f"p = {p_vel:.4f}", delta_color="off")
    m3.metric("Peso vs. Velocidad (r)", f"{r_pv:.4f}", f"p = {p_pv:.4f}", delta_color="off")

    st.info(
        f"**Hallazgo principal:** `Peso_Carga_Ton` tiene la relación lineal más fuerte con `Consumo_Litros` "
        f"($r = {r_peso:.4f}$, positiva muy fuerte). En cambio, la velocidad global presenta correlación lineal "
        f"prácticamente nula ($r = {r_vel:.4f}$), debido al efecto de confusión con el tipo de ruta."
    )

    mostrar_tendencias = st.checkbox("Mostrar rectas de tendencia por tipo de ruta", value=True)

    col_h, col_s = st.columns([0.9, 1.5])
    with col_h:
        _graficar_heatmap(df, corr_mat, num_cols)
    with col_s:
        _graficar_dispersion(df, rutas_sel, mostrar_tendencias)


def _graficar_heatmap(df: pd.DataFrame, corr_mat: pd.DataFrame, num_cols: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    labels_es = ["Peso\n(Ton)", "Velocidad\n(km/h)", "Consumo\n(Litros)"]
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels_es)
    ax.set_yticklabels(labels_es)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            r_ij = corr_mat.values[i, j]
            _, p_ij = stats.pearsonr(df[num_cols[i]], df[num_cols[j]])
            txt = f"{r_ij:.3f}\n(p<0.001)" if (i != j and p_ij < 0.001) else (
                f"{r_ij:.3f}\n(p={p_ij:.3f})" if i != j else "1.000"
            )
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if abs(r_ij) > 0.55 else "#111111",
                    fontsize=9, fontweight="bold")
    ax.set_title("Mapa de Calor (Pearson)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _graficar_dispersion(df: pd.DataFrame, rutas_sel: list[str], mostrar_tendencias: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), sharey=True)
    for ruta in rutas_sel:
        sub = df[df["Tipo_Ruta"] == ruta]
        axes[0].scatter(sub["Peso_Carga_Ton"], sub["Consumo_Litros"],
                        color=PALETTE[ruta], label=ruta, s=44, alpha=0.85, edgecolor="white")
        if mostrar_tendencias and len(sub) >= 2:
            m_r, b_r = np.polyfit(sub["Peso_Carga_Ton"], sub["Consumo_Litros"], 1)
            xl = np.linspace(sub["Peso_Carga_Ton"].min(), sub["Peso_Carga_Ton"].max(), 50)
            axes[0].plot(xl, m_r * xl + b_r, color=PALETTE[ruta], linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Peso de Carga (Toneladas)")
    axes[0].set_ylabel("Consumo de Combustible (Litros)")
    axes[0].set_title("Consumo vs. Peso de Carga")
    axes[0].legend(fontsize=8)

    for ruta in rutas_sel:
        sub = df[df["Tipo_Ruta"] == ruta]
        axes[1].scatter(sub["Velocidad_Promedio_kmh"], sub["Consumo_Litros"],
                        color=PALETTE[ruta], label=ruta, s=44, alpha=0.85, edgecolor="white")
        if mostrar_tendencias and len(sub) >= 2:
            m_v, b_v = np.polyfit(sub["Velocidad_Promedio_kmh"], sub["Consumo_Litros"], 1)
            xv = np.linspace(sub["Velocidad_Promedio_kmh"].min(), sub["Velocidad_Promedio_kmh"].max(), 50)
            axes[1].plot(xv, m_v * xv + b_v, color=PALETTE[ruta], linestyle="--", linewidth=1.4)
    axes[1].set_xlabel("Velocidad Promedio (km/h)")
    axes[1].set_title("Consumo vs. Velocidad Promedio")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
