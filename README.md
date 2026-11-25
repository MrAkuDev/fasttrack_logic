# FastTrack Logistics — Optimización de Rutas y Eficiencia de Combustible

Aplicación interactiva en **Streamlit** para el Proyecto Final (Caso 1) de la materia **Estadística Aplicada a la Investigación** (UACH — Facultad de Ingeniería, Maestría en Computación).

## Contenido del Dashboard
1. **Análisis Exploratorio y ANOVA:** Estadísticas descriptivas por tipo de ruta, boxplots, pruebas ANOVA de un factor ($F$, valor-$p$, $\eta^2$) y comparativa de medias.
2. **Análisis de Correlación:** Matriz de correlación de Pearson con valores-$p$, mapa de calor y diagramas de dispersión estratificados por ruta.
3. **Regresión Lineal Múltiple:** Ajuste OLS matricial, métricas de rendimiento ($R^2$, $MSE$, $RMSE$), pruebas $t$ de significancia y diagnóstico visual de residuos.
4. **Simulador de Consumo ("What-If"):** Controles interactivos para variar peso de carga, velocidad promedio, tipo de ruta y costo de diésel, con curvas de sensibilidad y cálculo de ahorro proyectado.
5. **Toma de Decisiones:** Respuestas ejecutivas basadas en evidencia estadística a las preguntas clave de FastTrack Logistics.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```
