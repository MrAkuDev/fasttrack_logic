# -*- coding: utf-8 -*-
"""Motor estadístico: carga de datos, descriptivas, ANOVA y regresión OLS."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """Carga el CSV desde la raíz del proyecto soportando cp850 y utf-8."""
    csv_path = Path(__file__).resolve().parent.parent / "datos_logistica_transporte.csv"
    try:
        return pd.read_csv(csv_path, encoding="cp850")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8")


def calcular_descriptivas(df: pd.DataFrame, col: str, rutas: list[str]) -> pd.DataFrame:
    """Calcula medidas de tendencia central y dispersión por tipo de ruta."""
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


def calcular_anova(df: pd.DataFrame, col: str, rutas: list[str]) -> dict | None:
    """Ejecuta ANOVA de un factor y devuelve tabla completa, F, valor-p y eta^2."""
    grupos = [df[df["Tipo_Ruta"] == r][col].values for r in rutas if len(df[df["Tipo_Ruta"] == r]) > 0]
    if len(grupos) < 2:
        return None
    n_total = sum(len(g) for g in grupos)
    k = len(grupos)
    gm = np.concatenate(grupos).mean()

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


def ajustar_ols(X_df: pd.DataFrame, y_ser: pd.Series) -> dict:
    """Ajusta un modelo de regresión lineal múltiple mediante OLS matricial."""
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


def predecir_con_ic(modelo: dict, x_vals: list[float]) -> dict:
    """Devuelve predicción puntual e intervalos de confianza/predicción al 95%."""
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


def entrenar_modelos_principales(df_raw: pd.DataFrame) -> tuple[dict, dict]:
    """Entrena el modelo base (Peso + Velocidad) y el modelo extendido (+ Tipo de Ruta)."""
    modelo_base = ajustar_ols(
        df_raw[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]],
        df_raw["Consumo_Litros"],
    )
    X_ext = pd.DataFrame({
        "Peso_Carga_Ton": df_raw["Peso_Carga_Ton"],
        "Velocidad_Promedio_kmh": df_raw["Velocidad_Promedio_kmh"],
        "Ruta_Urbana": (df_raw["Tipo_Ruta"] == "Urbana").astype(float),
        "Ruta_Montana": (df_raw["Tipo_Ruta"] == "Montaña").astype(float),
    })
    modelo_ext = ajustar_ols(X_ext, df_raw["Consumo_Litros"])
    return modelo_base, modelo_ext
