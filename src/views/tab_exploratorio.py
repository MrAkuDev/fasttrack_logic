# -*- coding: utf-8 -*-
"""Pestaña 1: Análisis Exploratorio y ANOVA de un Factor."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import PALETTE, VAR_MAP
from src.stats_engine import calcular_descriptivas, calcular_anova


def render_exploratorio(df: pd.DataFrame, rutas_sel: list[str]) -> None:
    st.subheader("1. Estadísticas Descriptivas por Tipo de Ruta")
    var_label = st.selectbox("Selecciona la variable a inspeccionar:", list(VAR_MAP.keys()))
    var_col = VAR_MAP[var_label]

    desc_df = calcular_descriptivas(df, var_col, rutas_sel)
    st.dataframe(
        desc_df.style.format({
            "Media": "{:.2f}", "Mediana": "{:.2f}", "Varianza": "{:.2f}",
            "Desv. Est.": "{:.2f}", "Mín.": "{:.2f}", "Máx.": "{:.2f}", "Rango": "{:.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    c_left, c_right = st.columns([1.25, 1.0])
    with c_left:
        _graficar_boxplot(df, var_col, var_label, rutas_sel)

    with c_right:
        st.markdown("#### Prueba de Hipótesis ANOVA de un Factor")
        st.latex(r"H_0: \mu_{\text{Autopista}} = \mu_{\text{Urbana}} = \mu_{\text{Montaña}} \quad (\alpha = 0.05)")
        anova_res = calcular_anova(df, var_col, rutas_sel)
        if anova_res is not None:
            a1, a2, a3 = st.columns(3)
            a1.metric("Estadístico F", f"{anova_res['F']:.4f}")
            a2.metric("Valor-p", f"{anova_res['p']:.3e}")
            a3.metric("Eta² (η²)", f"{anova_res['eta_sq']*100:.1f}%")

            st.dataframe(
                anova_res["tabla"].style.format({
                    "SC": "{:.4f}", "gl": "{:.0f}", "CM": "{:.4f}",
                    "F": "{:.4f}", "Valor-p": "{:.3e}",
                }, na_rep="—"),
                width="stretch",
                hide_index=True,
            )
            if anova_res["p"] < 0.05:
                st.success(
                    f"**Conclusión:** Como $p = {anova_res['p']:.3e} < 0.05$, se **rechaza $H_0$**. "
                    f"Existen diferencias estadísticamente significativas en **{var_label}** según el tipo de ruta."
                )
            else:
                st.info("No se rechaza $H_0$ con $\\alpha = 0.05$.")
        else:
            st.info("Selecciona al menos 2 tipos de ruta para ejecutar el ANOVA.")

    st.markdown("#### Resumen Comparativo ANOVA (las 3 variables cuantitativas)")
    if len(rutas_sel) >= 2:
        res_rows = []
        for lbl, cname in VAR_MAP.items():
            ar = calcular_anova(df, cname, rutas_sel)
            res_rows.append({
                "Variable": lbl,
                "Estadístico F": ar["F"],
                "Valor-p": f"{ar['p']:.3e}",
                "Varianza Explicada (η²)": f"{ar['eta_sq']*100:.2f}%",
                "Decisión (α = 0.05)": "Rechazar H₀" if ar["p"] < 0.05 else "No rechazar H₀",
            })
        st.dataframe(pd.DataFrame(res_rows), width="stretch", hide_index=True)

    with st.expander("Ver tabla de datos filtrados"):
        st.dataframe(df, width="stretch", hide_index=True)


def _graficar_boxplot(df: pd.DataFrame, var_col: str, var_label: str, rutas_sel: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    grupos_plot = [df[df["Tipo_Ruta"] == r][var_col].values for r in rutas_sel]
    bp = ax.boxplot(
        grupos_plot, positions=range(1, len(rutas_sel) + 1), widths=0.45,
        patch_artist=True, medianprops=dict(color="#111111", linewidth=2),
    )
    rng = np.random.default_rng(42)
    for i, (patch, ruta, g) in enumerate(zip(bp["boxes"], rutas_sel, grupos_plot), start=1):
        patch.set_facecolor(PALETTE[ruta])
        patch.set_alpha(0.35)
        patch.set_edgecolor(PALETTE[ruta])
        patch.set_linewidth(1.5)
        jitter = rng.normal(0, 0.05, size=len(g))
        ax.scatter(np.full(len(g), i) + jitter, g, color=PALETTE[ruta],
                   alpha=0.85, s=36, edgecolor="white", linewidth=0.5, zorder=3)
        m_val = float(np.mean(g))
        ax.scatter([i], [m_val], color="black", marker="D", s=45, zorder=4,
                   label="Media" if i == 1 else None)
        ax.text(i + 0.27, m_val, f"{m_val:.2f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(1, len(rutas_sel) + 1))
    ax.set_xticklabels([f"{r}\n(n={len(g)})" for r, g in zip(rutas_sel, grupos_plot)])
    ax.set_ylabel(var_label)
    ax.set_title(f"Distribución de {var_label} por Tipo de Ruta")
    ax.set_xlim(0.45, len(rutas_sel) + 0.65)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
