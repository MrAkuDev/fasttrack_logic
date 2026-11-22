# -*- coding: utf-8 -*-
"""
FastTrack Logistics — Dashboard de Optimización de Rutas y Eficiencia de Combustible
Proyecto Final 1 | Estadística Aplicada a la Investigación (UACH)
Alumno: Joel Arturo Álvarez Hernández (Matrícula: 403233)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FastTrack Logistics | Eficiencia de Combustible",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 130,
})

PALETTE = {
    "Autopista": "#1f77b4",
    "Urbana": "#ff7f0e",
    "Montaña": "#d62728",
}
ORDER = ["Autopista", "Urbana", "Montaña"]

# -----------------------------------------------------------------------------
# FUNCIONES DE CARGA Y MODELADO ESTADÍSTICO
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "datos_logistica_transporte.csv"
    try:
        df = pd.read_csv(csv_path, encoding="cp850")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")
    return df


def calcular_descriptivas(df: pd.DataFrame, col: str, rutas: list[str]) -> pd.DataFrame:
    filas = []
    for r in rutas:
        g = df[df["Tipo_Ruta"] == r][col]
        if len(g) == 0:
            continue
        vc = g.value_counts()
        modas = vc[vc > 1].index.tolist()
        moda_str = ", ".join(f"{m:.1f}" for m in modas) if modas else "Sin moda"
        filas.append({
            "Tipo de Ruta": r,
            "n": int(len(g)),
            "Media": g.mean(),
            "Mediana": g.median(),
            "Varianza": g.var(ddof=1) if len(g) > 1 else 0.0,
            "Desv. Est.": g.std(ddof=1) if len(g) > 1 else 0.0,
            "Mín.": g.min(),
            "Máx.": g.max(),
            "Rango": g.max() - g.min(),
            "Moda": moda_str,
        })
    return pd.DataFrame(filas)


def calcular_anova(df: pd.DataFrame, col: str, rutas: list[str]):
    grupos = [df[df["Tipo_Ruta"] == r][col].values for r in rutas if len(df[df["Tipo_Ruta"] == r]) > 0]
    if len(grupos) < 2:
        return None
    n_total = sum(len(g) for g in grupos)
    k = len(grupos)
    all_vals = np.concatenate(grupos)
    gm = all_vals.mean()
    sc_entre = float(sum(len(g) * (g.mean() - gm) ** 2 for g in grupos))
    sc_dentro = float(sum(np.sum((g - g.mean()) ** 2) for g in grupos))
    sc_total = sc_entre + sc_dentro
    gl_e, gl_d = k - 1, n_total - k
    cm_entre = sc_entre / gl_e
    cm_dentro = sc_dentro / gl_d
    f_stat, p_val = stats.f_oneway(*grupos)
    eta_sq = sc_entre / sc_total if sc_total > 0 else 0.0
    tabla = pd.DataFrame({
        "Fuente": ["Entre rutas", "Dentro de rutas", "Total"],
        "SC": [sc_entre, sc_dentro, sc_total],
        "gl": [gl_e, gl_d, n_total - 1],
        "CM": [cm_entre, cm_dentro, np.nan],
        "F": [float(f_stat), np.nan, np.nan],
        "Valor-p": [float(p_val), np.nan, np.nan],
    })
    return {
        "tabla": tabla,
        "F": float(f_stat),
        "p": float(p_val),
        "eta_sq": float(eta_sq),
        "gl_e": gl_e,
        "gl_d": gl_d,
    }


def ajustar_ols(X_df: pd.DataFrame, y_ser: pd.Series):
    n = len(y_ser)
    nombres = ["Intercepto"] + list(X_df.columns)
    X = np.column_stack([np.ones(n), X_df.values.astype(float)])
    Y = y_ser.values.astype(float).reshape(-1, 1)
    k = X.shape[1] - 1
    gl_res = n - k - 1

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = (XtX_inv @ (X.T @ Y)).flatten()

    y_pred = (X @ beta.reshape(-1, 1)).flatten()
    residuos = y_ser.values - y_pred

    sc_tot = float(np.sum((y_ser.values - y_ser.mean()) ** 2))
    sc_res = float(np.sum(residuos ** 2))
    sc_reg = sc_tot - sc_res
    mse = sc_res / n
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuos)))
    r2 = float(1.0 - sc_res / sc_tot)
    r2_adj = float(1.0 - (1.0 - r2) * (n - 1) / gl_res)

    cm_res = sc_res / gl_res
    f_global = float((sc_reg / k) / cm_res)
    p_global = float(stats.f.sf(f_global, k, gl_res))

    se_beta = np.sqrt(np.diag(cm_res * XtX_inv))
    t_stats = beta / se_beta
    p_vals = 2 * stats.t.sf(np.abs(t_stats), gl_res)
    t_crit = float(stats.t.ppf(0.975, gl_res))

    # Coeficientes estandarizados
    std_y = np.std(y_ser.values, ddof=1)
    beta_std = [np.nan]
    for idx, col in enumerate(X_df.columns, start=1):
        std_x = np.std(X_df[col].values, ddof=1)
        beta_std.append(float(beta[idx] * std_x / std_y))

    tabla_coefs = pd.DataFrame({
        "Parámetro": nombres,
        "Coeficiente (β)": beta,
        "Error Est.": se_beta,
        "Estadístico t": t_stats,
        "Valor-p": p_vals,
        "β Estandarizado": beta_std,
        "Significativo (α=0.05)": ["Sí (Rechazar H₀)" if p < 0.05 else "No" for p in p_vals],
    })

    return {
        "beta": beta,
        "XtX": XtX,
        "XtX_inv": XtX_inv,
        "cm_res": float(cm_res),
        "t_crit": t_crit,
        "gl_res": gl_res,
        "y_pred": y_pred,
        "residuos": residuos,
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_adj": r2_adj,
        "f_global": f_global,
        "p_global": p_global,
        "tabla_coefs": tabla_coefs,
    }


def predecir_con_ic(modelo: dict, x_vals: list[float]):
    x_vec = np.array([1.0] + x_vals, dtype=float)
    y_hat = float(x_vec @ modelo["beta"])
    h0 = float(x_vec @ modelo["XtX_inv"] @ x_vec)
    se_mean = float(np.sqrt(modelo["cm_res"] * h0))
    se_pred = float(np.sqrt(modelo["cm_res"] * (1.0 + h0)))
    t_c = modelo["t_crit"]
    return {
        "pred": y_hat,
        "ci_low": y_hat - t_c * se_mean,
        "ci_high": y_hat + t_c * se_mean,
        "pi_low": y_hat - t_c * se_pred,
        "pi_high": y_hat + t_c * se_pred,
    }


# -----------------------------------------------------------------------------
# CARGA DE DATOS Y MODELOS COMPLETOS
# -----------------------------------------------------------------------------
df_raw = cargar_datos()

# Modelos entrenados sobre el dataset completo (para consistencia en simulador y decisiones)
modelo_base = ajustar_ols(df_raw[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]], df_raw["Consumo_Litros"])

X_ext = pd.DataFrame({
    "Peso_Carga_Ton": df_raw["Peso_Carga_Ton"],
    "Velocidad_Promedio_kmh": df_raw["Velocidad_Promedio_kmh"],
    "Ruta_Urbana": (df_raw["Tipo_Ruta"] == "Urbana").astype(float),
    "Ruta_Montana": (df_raw["Tipo_Ruta"] == "Montaña").astype(float),
})
modelo_ext = ajustar_ols(X_ext, df_raw["Consumo_Litros"])

# -----------------------------------------------------------------------------
# BARRA LATERAL (FILTROS Y CONTEXTO)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚛 FastTrack Logistics")
    st.caption("Proyecto Final 1 — Estadística Aplicada a la Investigación (UACH)")
    st.divider()

    st.markdown("### Filtros de Exploración")
    rutas_sel = st.multiselect(
        "Tipos de ruta a incluir:",
        options=ORDER,
        default=ORDER,
    )
    peso_min, peso_max = float(df_raw["Peso_Carga_Ton"].min()), float(df_raw["Peso_Carga_Ton"].max())
    rango_peso = st.slider(
        "Rango de peso de carga (Ton):",
        min_value=float(np.floor(peso_min)),
        max_value=float(np.ceil(peso_max)),
        value=(float(np.floor(peso_min)), float(np.ceil(peso_max))),
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

# -----------------------------------------------------------------------------
# ENCABEZADO PRINCIPAL Y KPIs
# -----------------------------------------------------------------------------
st.title("Optimización de Rutas y Eficiencia de Combustible")
st.markdown(
    "Diagnóstico estadístico inferencial y modelado predictivo para mitigar el **incremento del 18%** "
    "en costos operativos por consumo de diésel en la flota de **FastTrack Logistics**."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Viajes analizados", f"{len(df)} / {len(df_raw)}")
k2.metric("Consumo promedio", f"{df['Consumo_Litros'].mean():.2f} L", f"DE: {df['Consumo_Litros'].std(ddof=1):.2f} L")
k3.metric("Peso promedio", f"{df['Peso_Carga_Ton'].mean():.2f} Ton", f"Máx: {df['Peso_Carga_Ton'].max():.1f} t")
k4.metric("Velocidad promedio", f"{df['Velocidad_Promedio_kmh'].mean():.1f} km/h")

st.divider()

# -----------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploratorio y ANOVA",
    "🔗 Correlación",
    "📈 Regresión Lineal Múltiple",
    "🎛️ Simulador de Consumo",
    "🎯 Toma de Decisiones",
])

# =============================================================================
# TAB 1: ANÁLISIS EXPLORATORIO Y ANOVA
# =============================================================================
with tab1:
    st.subheader("1. Estadísticas Descriptivas por Tipo de Ruta")
    var_map = {
        "Consumo de Combustible (Litros)": "Consumo_Litros",
        "Peso de Carga (Toneladas)": "Peso_Carga_Ton",
        "Velocidad Promedio (km/h)": "Velocidad_Promedio_kmh",
    }
    var_label = st.selectbox("Selecciona la variable a inspeccionar:", list(var_map.keys()))
    var_col = var_map[var_label]

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
        for lbl, cname in var_map.items():
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

# =============================================================================
# TAB 2: ANÁLISIS DE CORRELACIÓN
# =============================================================================
with tab2:
    st.subheader("2. Matriz de Correlación de Pearson y Dispersión")
    num_cols = ["Peso_Carga_Ton", "Velocidad_Promedio_kmh", "Consumo_Litros"]
    corr_mat = df[num_cols].corr(method="pearson")

    r_peso, p_peso = stats.pearsonr(df["Peso_Carga_Ton"], df["Consumo_Litros"])
    r_vel, p_vel = stats.pearsonr(df["Velocidad_Promedio_kmh"], df["Consumo_Litros"])
    r_pv, p_pv = stats.pearsonr(df["Peso_Carga_Ton"], df["Velocidad_Promedio_kmh"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Peso vs. Consumo (r)", f"{r_peso:.4f}", f"p = {p_peso:.2e}")
    m2.metric("Velocidad vs. Consumo (r)", f"{r_vel:.4f}", f"p = {p_vel:.4f}")
    m3.metric("Peso vs. Velocidad (r)", f"{r_pv:.4f}", f"p = {p_pv:.4f}")

    st.info(
        f"**Hallazgo principal:** `Peso_Carga_Ton` tiene la relación lineal más fuerte con `Consumo_Litros` "
        f"($r = {r_peso:.4f}$, positiva muy fuerte). En cambio, la velocidad global presenta correlación lineal "
        f"prácticamente nula ($r = {r_vel:.4f}$), debido al efecto de confusión con el tipo de ruta."
    )

    mostrar_tendencias = st.checkbox("Mostrar rectas de tendencia por tipo de ruta", value=True)

    col_h, col_s = st.columns([0.9, 1.5])
    with col_h:
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

    with col_s:
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

# =============================================================================
# TAB 3: REGRESIÓN LINEAL MÚLTIPLE
# =============================================================================
with tab3:
    st.subheader("3. Regresión Lineal Múltiple (Mínimos Cuadrados Ordinarios)")
    incluir_ruta = st.toggle(
        "Comparar con Modelo Extendido (incorporando Tipo de Ruta como variables dummy)",
        value=False,
    )

    mod_activo = modelo_ext if incluir_ruta else modelo_base
    b = mod_activo["beta"]

    if not incluir_ruta:
        st.latex(
            rf"\widehat{{Y}} = {b[0]:.4f} + {b[1]:.4f}\,X_1 \;-\; {abs(b[2]):.4f}\,X_2"
        )
        st.caption("Donde $X_1$ = Peso de carga (Ton) y $X_2$ = Velocidad promedio (km/h). Estimado sobre los 50 viajes.")
    else:
        st.latex(
            rf"\widehat{{Y}} = {b[0]:.4f} + {b[1]:.4f}\,X_1 + {b[2]:.4f}\,X_2 "
            rf"+ {b[3]:.4f}\,\mathbb{{I}}(\text{{Urbana}}) + {b[4]:.4f}\,\mathbb{{I}}(\text{{Montaña}})"
        )
        st.caption("Categoría de referencia: **Autopista**. Incorporar el tipo de ruta eleva $R^2$ de 80.98% a 97.48%.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("R² (Determinación)", f"{mod_activo['r2']*100:.2f}%", f"R² ajust.: {mod_activo['r2_adj']*100:.2f}%")
    r2.metric("RMSE", f"{mod_activo['rmse']:.2f} L", f"MAE: {mod_activo['mae']:.2f} L")
    r3.metric("MSE", f"{mod_activo['mse']:.2f} L²")
    r4.metric("F Global", f"{mod_activo['f_global']:.2f}", f"p = {mod_activo['p_global']:.2e}")

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

# =============================================================================
# TAB 4: SIMULADOR INTERACTIVO DE CONSUMO ("WHAT-IF")
# =============================================================================
with tab4:
    st.subheader("4. Simulador Interactivo de Consumo y Costos Operativos")
    st.markdown(
        "Modifica el peso de la carga, la velocidad promedio, el tipo de ruta y el precio del diésel para "
        "explorar distintos escenarios operativos en tiempo real."
    )

    s_col1, s_col2 = st.columns([1.0, 1.35])

    with s_col1:
        st.markdown("#### Parámetros del Viaje")
        sim_peso = st.slider("Peso de carga (Toneladas)", min_value=4.0, max_value=30.0, value=15.0, step=0.5)
        sim_vel = st.slider("Velocidad promedio (km/h)", min_value=20.0, max_value=110.0, value=80.0, step=1.0)
        sim_ruta = st.selectbox("Tipo de ruta (para estimación ajustada)", options=ORDER, index=0)

        st.markdown("#### Parámetros Financieros")
        precio_diesel = st.number_input("Precio del diésel (MXN / Litro)", min_value=10.0, max_value=50.0, value=25.50, step=0.50)
        viajes_mes = st.slider("Viajes mensuales estimados", min_value=10, max_value=500, value=100, step=10)

    # Cálculos del simulador
    pred_base = predecir_con_ic(modelo_base, [sim_peso, sim_vel])
    d_urb = 1.0 if sim_ruta == "Urbana" else 0.0
    d_mon = 1.0 if sim_ruta == "Montaña" else 0.0
    pred_ruta = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, d_urb, d_mon])

    # Comparación entre las 3 rutas para el mismo peso y velocidad
    pred_auto = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, 0.0, 0.0])["pred"]
    pred_urb = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, 1.0, 0.0])["pred"]
    pred_mon = predecir_con_ic(modelo_ext, [sim_peso, sim_vel, 0.0, 1.0])["pred"]

    with s_col2:
        st.markdown("#### Resultados del Escenario Simulado")
        res1, res2, res3 = st.columns(3)
        res1.metric(
            "Consumo (Modelo Base)",
            f"{pred_base['pred']:.2f} L",
            f"IC 95%: [{pred_base['ci_low']:.1f}, {pred_base['ci_high']:.1f}] L",
        )
        res2.metric(
            f"Consumo ({sim_ruta})",
            f"{pred_ruta['pred']:.2f} L",
            f"Costo/viaje: ${pred_ruta['pred'] * precio_diesel:,.0f} MXN",
        )
        ahorro_vs_mon = max(0.0, pred_mon - pred_auto) * precio_diesel * viajes_mes
        res3.metric(
            "Ahorro Autopista vs. Montaña",
            f"${ahorro_vs_mon:,.0f} MXN/mes",
            f"{pred_mon - pred_auto:.1f} L menos por viaje",
        )

        # Gráficas de sensibilidad y comparación de rutas
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

        # Panel A: Sensibilidad respecto al Peso (manteniendo sim_vel constante)
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
        axes[0].scatter([sim_peso], [pred_base["pred"]], color="#333333", s=65, zorder=5)
        axes[0].scatter([sim_peso], [pred_ruta["pred"]], color=PALETTE[sim_ruta], s=75, zorder=5, edgecolor="black")
        axes[0].set_xlabel("Peso de Carga (Toneladas)")
        axes[0].set_ylabel("Consumo Estimado (Litros)")
        axes[0].set_title(f"Sensibilidad al Peso (a {sim_vel:.0f} km/h)")
        axes[0].legend(fontsize=8)

        # Panel B: Comparativa de las 3 rutas para el peso/velocidad actuales
        rutas_vals = [pred_auto, pred_urb, pred_mon]
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

# =============================================================================
# TAB 5: TOMA DE DECISIONES
# =============================================================================
with tab5:
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
