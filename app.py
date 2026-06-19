import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Valor de Vivienda",
    page_icon="🏠",
    layout="wide",
)

# ── Estilos personalizados ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F3F8; }
    .block-container { padding-top: 2rem; }
    .pred-box {
        background: linear-gradient(135deg, #E50914 0%, #b00710 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .pred-label { font-size: 0.85rem; font-weight: 800; letter-spacing: 0.15em;
                  text-transform: uppercase; opacity: 0.8; margin-bottom: 0.5rem; }
    .pred-value { font-size: 3rem; font-weight: 900; line-height: 1; }
    .pred-sub   { font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem; }
    .info-card  {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #E50914;
    }
    .section-title {
        font-size: 0.7rem; font-weight: 900; letter-spacing: 0.15em;
        text-transform: uppercase; color: #94a3b8; margin-bottom: 0.75rem;
    }
    div[data-testid="stSlider"] > div { padding-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)


# ── Carga del modelo ──────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """
    Busca el modelo entrenado en la misma carpeta.
    Acepta archivos .pkl o .joblib.
    """
    for nombre in ["modelo.pkl", "model.pkl", "modelo.joblib", "model.joblib"]:
        if os.path.exists(nombre):
            return joblib.load(nombre), nombre
    return None, None

modelo, nombre_archivo = cargar_modelo()


# ── Encabezado ────────────────────────────────────────────────────────────
st.markdown("## 🏠 Predictor de Valor de Vivienda")
st.markdown(
    "Ajusta las variables del inmueble y obtén la predicción del "
    "**valor_mediano_vivienda** generada por el modelo de Machine Learning."
)

if modelo is None:
    st.error(
        "⚠️ No se encontró ningún archivo de modelo (.pkl o .joblib) en esta carpeta. "
        "Asegúrate de que el archivo esté en el mismo directorio que app.py."
    )
    st.info(
        "Para guardar tu modelo desde el notebook usa:\n"
        "```python\nimport joblib\njoblib.dump(mi_modelo, 'modelo.pkl')\n```"
    )
    st.stop()
else:
    st.success(f"✅ Modelo cargado correctamente: `{nombre_archivo}`")

st.markdown("---")


# ── Panel de variables ────────────────────────────────────────────────────
col_izq, col_der = st.columns([3, 2], gap="large")

with col_izq:

    # — Ubicación geográfica —
    st.markdown('<p class="section-title">📍 Ubicación geográfica</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        longitud = st.slider(
            "Longitud",
            min_value=-124.35, max_value=-114.31,
            value=-119.57, step=0.01,
            help="Coordenada de longitud del bloque. California va de -124° a -114°"
        )
    with c2:
        latitud = st.slider(
            "Latitud",
            min_value=32.54, max_value=41.95,
            value=35.63, step=0.01,
            help="Coordenada de latitud del bloque. California va de 32° a 42°"
        )

    st.markdown("---")

    # — Características del bloque —
    st.markdown('<p class="section-title">🏘️ Características del bloque</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        edad_mediana_vivienda = st.slider(
            "Edad mediana de vivienda (años)",
            min_value=1, max_value=52,
            value=29,
            help="Antigüedad mediana de las viviendas del bloque"
        )
        total_habitaciones = st.number_input(
            "Total de habitaciones",
            min_value=2, max_value=39320,
            value=2635,
            step=50,
            help="Suma de todas las habitaciones del bloque"
        )
        total_dormitorios = st.number_input(
            "Total de dormitorios",
            min_value=1, max_value=6445,
            value=537,
            step=10,
            help="Suma de todos los dormitorios del bloque"
        )
    with c4:
        poblacion = st.number_input(
            "Población",
            min_value=3, max_value=35682,
            value=1425,
            step=50,
            help="Número de personas que habitan el bloque"
        )
        hogares = st.number_input(
            "Hogares",
            min_value=1, max_value=6082,
            value=499,
            step=10,
            help="Número de hogares en el bloque"
        )
        ingreso_mediano = st.slider(
            "Ingreso mediano (x$10,000 USD)",
            min_value=0.5, max_value=15.0,
            value=3.87, step=0.01,
            help="Ingreso mediano de los hogares del bloque (en decenas de miles de dólares)"
        )

    st.markdown("---")

    # — Proximidad al océano —
    st.markdown('<p class="section-title">🌊 Proximidad al océano</p>', unsafe_allow_html=True)
    proximidad_oceano = st.selectbox(
        "Categoría de proximidad",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        index=0,
        help="Clasificación geográfica del bloque respecto al océano Pacífico"
    )

    descripciones_oceano = {
        "<1H OCEAN":  "A menos de 1 hora del océano en auto",
        "INLAND":     "Tierra adentro, lejos del océano",
        "ISLAND":     "En una isla",
        "NEAR BAY":   "Cerca de la bahía de San Francisco",
        "NEAR OCEAN": "Cerca del océano Pacífico",
    }
    st.caption(f"ℹ️ {descripciones_oceano[proximidad_oceano]}")


# ── Resumen de variables y predicción ────────────────────────────────────
with col_der:

    st.markdown('<p class="section-title">📋 Resumen de entrada</p>', unsafe_allow_html=True)

    resumen = {
        "Longitud":                    f"{longitud:.2f}",
        "Latitud":                     f"{latitud:.2f}",
        "Edad mediana vivienda":       f"{edad_mediana_vivienda} años",
        "Total habitaciones":          f"{total_habitaciones:,}",
        "Total dormitorios":           f"{total_dormitorios:,}",
        "Población":                   f"{poblacion:,}",
        "Hogares":                     f"{hogares:,}",
        "Ingreso mediano":             f"${ingreso_mediano:.2f} × $10K",
        "Proximidad al océano":        proximidad_oceano,
    }

    for etiqueta, valor in resumen.items():
        st.markdown(
            f'<div class="info-card">'
            f'<span style="font-size:0.72rem;color:#94a3b8;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.08em">{etiqueta}</span><br>'
            f'<span style="font-size:1rem;font-weight:800;color:#1e293b">{valor}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Botón de predicción ───────────────────────────────────────────────
    if st.button("🔮 Generar predicción", use_container_width=True, type="primary"):

        # Construir el dataframe de entrada
        datos_entrada = pd.DataFrame([{
            "longitude":          longitud,
            "latitude":           latitud,
            "housing_median_age": edad_mediana_vivienda,
            "total_rooms":        total_habitaciones,
            "total_bedrooms":     total_dormitorios,
            "population":         poblacion,
            "households":         hogares,
            "median_income":      ingreso_mediano,
            "ocean_proximity":    proximidad_oceano,
        }])

        try:
            prediccion = modelo.predict(datos_entrada)[0]
            prediccion_formateada = f"${prediccion:,.0f}"

            st.markdown(
                f'<div class="pred-box">'
                f'<div class="pred-label">Valor mediano estimado</div>'
                f'<div class="pred-value">{prediccion_formateada}</div>'
                f'<div class="pred-sub">USD — Predicción del modelo de ML</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Interpretación automática
            st.markdown("#### 📊 Interpretación")
            if prediccion < 100_000:
                nivel = "🟢 Zona de bajo costo"
                desc  = "El valor estimado está por debajo del promedio del mercado californiano."
            elif prediccion < 250_000:
                nivel = "🟡 Zona de costo moderado"
                desc  = "El valor estimado se encuentra en el rango medio del mercado."
            elif prediccion < 400_000:
                nivel = "🟠 Zona de alto costo"
                desc  = "El valor estimado supera la media estatal de California."
            else:
                nivel = "🔴 Zona de costo muy alto"
                desc  = "El valor estimado está en el segmento premium del mercado."

            st.info(f"**{nivel}** — {desc}")

        except Exception as e:
            st.error(
                f"Error al generar la predicción: `{e}`\n\n"
                "Verifica que las columnas del modelo coincidan con las variables de entrada, "
                "especialmente el encoding de `ocean_proximity`."
            )

    else:
        st.markdown(
            '<div style="text-align:center;padding:2rem;color:#94a3b8;">'
            '<div style="font-size:2rem">🔮</div>'
            '<div style="font-size:0.85rem;font-weight:700;margin-top:0.5rem">'
            'Ajusta las variables y presiona<br>el botón para ver la predicción'
            '</div></div>',
            unsafe_allow_html=True
        )

# ── Pie de página ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🎓 Ejercicio de modelo predictivo — California Housing Dataset | "
    "Variables: longitude, latitude, housing_median_age, total_rooms, "
    "total_bedrooms, population, households, median_income, ocean_proximity → "
    "Target: median_house_value"
)
