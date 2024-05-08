#%%
# Importar librerías necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#%%
# Configuración de la página
st.set_page_config(page_title="Análisis de Comunas en Región Metropolitana", layout='wide')

# Incluir logo
logo_url = "https://upload.wikimedia.org/wikipedia/commons/6/6e/SEREMISALUDMET.png"
st.image(logo_url, width=200)  # Ajusta el ancho según tus necesidades

st.title("Tablero de Control para las Comunas de la Región Metropolitana")

# Descripción del tablero
st.write("""
### Bienvenido al Tablero Interactivo de Comunas

Este tablero está diseñado para proporcionar una visión integral y detallada de los diversos indicadores socioeconómicos, demográficos y urbanos de las comunas que forman la Región Metropolitana de Santiago. Aquí podrás explorar y visualizar datos que abarcan desde distribuciones de población y niveles de ingresos hasta aspectos de salud y educación.

Utiliza los menús desplegables para seleccionar diferentes comunas y visualizar datos específicos relacionados con cada una. El tablero está enriquecido con gráficos interactivos que te ayudarán a entender mejor las características y desafíos de cada comuna.

### Cómo usar este tablero

- **Selecciona una comuna:** Utiliza el menú desplegable para elegir una comuna y automáticamente se actualizarán los gráficos y tablas para reflejar los datos correspondientes.
- **Explora los gráficos:** Interactúa con los gráficos para obtener detalles específicos sobre diferentes indicadores.
- **Comparación y análisis:** Compara datos entre diferentes comunas ajustando tu selección y analiza las tendencias y patrones que emergen de los datos.

### Objetivos del tablero

- **Informar:** Proporcionar datos actualizados y precisos sobre las comunas.
- **Analizar:** Facilitar el análisis comparativo entre comunas.
- **Impulsar decisiones:** Ayudar a tomadores de decisiones y ciudadanos a entender las dinámicas comunitarias para una planificación y desarrollo mejor informado.

Esperamos que este tablero sea una herramienta útil para todos los interesados en el desarrollo urbano y social de la Región Metropolitana de Santiago. ¡Explora, descubre e interactúa!

""")

# st.sidebar.title("Navegación")
# opcion = st.sidebar.radio(
#     "Seleccione un tópico:",
#     ('Territorio y Demografía', 'Pirámide Poblacional', 'Socioeconómico', 
#      'Mortalidad General', 'Mortalidad GG', 'Mortalidad CE', 'Estilos Nutricional', 
#      'Estilos de vida y FR', 'Estilos de vida y FR (2)', 'PNI', 'Salud', 'Mortalidad Infantil',
#      'Pobreza', 'Diarréa', 'Egresos Hospitalarios')
# )

# # Configuración inicial del encabezado y descripción
# st.title("Tablero de Control para las Comunas de la Región Metropolitana")
# st.markdown("### Explore los diferentes aspectos de las comunas a través de los datos y visualizaciones interactivas.")



