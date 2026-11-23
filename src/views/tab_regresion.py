# -*- coding: utf-8 -*-
"""Pestaña 3: Regresión Lineal Múltiple y Diagnóstico de Residuos."""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import ORDER, PALETTE


def render_regresion(df_raw: pd.DataFrame, modelo_base: dict, modelo_ext: dict) -> None:
    st.subheader("3. Regresión Lineal Múltiple (Mínimos Cuadrados Ordinarios)")
    incluir_ruta = st.toggle(
        "Comparar con Modelo Extendido (incorporando Tipo de Ruta como variables dummy)",
        value=False,
    )

    mod_activo = modelo_ext if incluir_ruta else modelo_base
    b = mod_activo["beta"]

    if not incluir_ruta:
        st.latex(rf"\widehat{{Y}} = {b[0]:.4f} + {b[1]:.4f}\,X_1 \;-\; {abs(b[2]):.4f}\,X_2")
        st.caption("Donde $X_1$ = Peso de carga (Ton) y $X_2$ = Velocidad promedio (km/h). Estimado sobre los 50 viajes.")
    else:
        st.latex(
            rf"\widehat{{Y}} = {b[0]:.4f} + {b[1]:.4f}\,X_1 + {b[2]:.4f}\,X_2 "
            rf"+ {b[3]:.4f}\,\mathbb{{I}}(\text{{Urbana}}) + {b[4]:.4f}\,\mathbb{{I}}(\text{{Montaña}})"
        )
        st.caption("Categoría de referencia: **Autopista**. Incorporar el tipo de ruta eleva $R^2$ de 80.98% a 97.48%.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("R² (Determinación)", f"{mod_activo['r2']*100:.2f}%", f"R² ajust.: {mod_activo['r2_adj']*100:.2f}%", delta_color="off")
    r2.metric("RMSE", f"{mod_activo['rmse']:.2f} L", f"MAE: {mod_activo['mae']:.2f} L", delta_color="off")
    r3.metric("MSE", f"{mod_activo['mse']:.2f} L²")
    r4.metric("F Global", f"{mod_activo['f_global']:.2f}", f"p = {mod_activo['p_global']:.2e}", delta_color="off")

    st.markdown("#### Validación de Significancia de Coeficientes (Pruebas $t$, $\\alpha = 0.05$)")
    st.dataframe(
        mod_activo["tabla_coefs"].style.format({
            "Coeficiente (β)": "{:.4f}",
            "Error Est.": "{:.4f}",
            "Estadístico t": "{:.4f}",
            "Valor-p": "{:.3e}",
            "β Estandarizado": "{:.4f}",
        }, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    _graficar_diagnostico(df_raw, mod_activo)


def _graficar_diagnostico(df_raw: pd.DataFrame, mod_activo: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    y_true = df_raw["Consumo_Litros"].values
    for ruta in ORDER:
        mask = (df_raw["Tipo_Ruta"] == ruta).values
        axes[0].scatter(y_true[mask], mod_activo["y_pred"][mask], color=PALETTE[ruta],
                        label=ruta, s=42, alpha=0.85, edgecolor="white")
    lims = [20, 96]
    axes[0].plot(lims, lims, "k--", linewidth=1.3)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Consumo Real (Litros)")
    axes[0].set_ylabel("Consumo Predicho (Litros)")
    axes[0].set_title(f"Real vs. Predicho (R² = {mod_activo['r2']:.4f}, RMSE = {mod_activo['rmse']:.2f} L)")
    axes[0].legend()

    for ruta in ORDER:
        mask = (df_raw["Tipo_Ruta"] == ruta).values
        axes[1].scatter(mod_activo["y_pred"][mask], mod_activo["residuos"][mask], color=PALETTE[ruta],
                        label=ruta, s=42, alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.3)
    axes[1].set_xlabel("Consumo Predicho (Litros)")
    axes[1].set_ylabel("Residuo (Litros)")
    axes[1].set_title("Residuos vs. Valores Ajustados")
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
