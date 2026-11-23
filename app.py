# -*- coding: utf-8 -*-
"""
FastTrack Logistics — Dashboard de Optimización de Rutas y Eficiencia de Combustible
Proyecto Final 1 | Estadística Aplicada a la Investigación (UACH)
Alumno: Joel Arturo Álvarez Hernández (Matrícula: 403233)
"""

import numpy as np
import streamlit as st

from src.config import ORDER, configurar_matplotlib
from src.stats_engine import cargar_datos, entrenar_modelos_principales
from src.views.tab_exploratorio import render_exploratorio
from src.views.tab_correlacion import render_correlacion
from src.views.tab_regresion import render_regresion
from src.views.tab_simulador import render_simulador
from src.views.tab_decisiones import render_decisiones


def main() -> None:
    st.set_page_config(
        page_title="FastTrack Logistics | Eficiencia de Combustible",
        page_icon="🚛",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configurar_matplotlib()

    # Carga de datos y entrenamiento de modelos OLS sobre el dataset completo
    df_raw = cargar_datos()
    modelo_base, modelo_ext = entrenar_modelos_principales(df_raw)

    # Barra lateral con filtros
    with st.sidebar:
        st.markdown("## 🚛 FastTrack Logistics")
        st.caption("Proyecto Final 1 — Estadística Aplicada a la Investigación (UACH)")
        st.divider()

        st.markdown("### Filtros de Exploración")
        rutas_sel = st.multiselect("Tipos de ruta a incluir:", options=ORDER, default=ORDER)
        peso_min = float(np.floor(df_raw["Peso_Carga_Ton"].min()))
        peso_max = float(np.ceil(df_raw["Peso_Carga_Ton"].max()))
        rango_peso = st.slider(
            "Rango de peso de carga (Ton):",
            min_value=peso_min,
            max_value=peso_max,
            value=(peso_min, peso_max),
            step=0.5,
        )

        st.divider()
        st.markdown(
            "**Alumno:** Joel Arturo Álvarez Hernández  \n"
            "**Matrícula:** 403233  \n"
            "**Dataset:** `datos_logistica_transporte.csv` ($N=50$)"
        )

    if not rutas_sel:
        st.warning("Selecciona al menos un tipo de ruta en la barra lateral.")
        st.stop()

    df = df_raw[
        (df_raw["Tipo_Ruta"].isin(rutas_sel))
        & (df_raw["Peso_Carga_Ton"] >= rango_peso[0])
        & (df_raw["Peso_Carga_Ton"] <= rango_peso[1])
    ].copy()

    if len(df) < 3:
        st.warning("El filtro actual deja muy pocos registros. Amplía el rango en la barra lateral.")
        st.stop()

    # Encabezado y KPIs globales
    st.title("Optimización de Rutas y Eficiencia de Combustible")
    st.markdown(
        "Diagnóstico estadístico inferencial y modelado predictivo para mitigar el **incremento del 18%** "
        "en costos operativos por consumo de diésel en la flota de **FastTrack Logistics**."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Viajes analizados", f"{len(df)} / {len(df_raw)}")
    k2.metric("Consumo promedio", f"{df['Consumo_Litros'].mean():.2f} L", f"DE: {df['Consumo_Litros'].std(ddof=1):.2f} L", delta_color="off")
    k3.metric("Peso promedio", f"{df['Peso_Carga_Ton'].mean():.2f} Ton", f"Máx: {df['Peso_Carga_Ton'].max():.1f} t", delta_color="off")
    k4.metric("Velocidad promedio", f"{df['Velocidad_Promedio_kmh'].mean():.1f} km/h")
    st.divider()

    # Pestañas modulares
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Exploratorio y ANOVA",
        "🔗 Correlación",
        "📈 Regresión Lineal Múltiple",
        "🎛️ Simulador de Consumo",
        "🎯 Toma de Decisiones",
    ])

    with tab1:
        render_exploratorio(df, rutas_sel)
    with tab2:
        render_correlacion(df, rutas_sel)
    with tab3:
        render_regresion(df_raw, modelo_base, modelo_ext)
    with tab4:
        render_simulador(modelo_base, modelo_ext)
    with tab5:
        render_decisiones(df_raw, modelo_base, modelo_ext)


if __name__ == "__main__":
    main()
