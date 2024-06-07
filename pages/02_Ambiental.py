#%%
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from shapely.geometry import Point

# Listado comunas
lista_comunas = [
    'Todas las comunas', 'Alhué', 'Buin', 'Calera de Tango', 'Cerrillos', 'Cerro Navia', 'Colina', 
    'Conchalí', 'Curacaví', 'El Bosque', 'El Monte', 'Estación Central', 'Huechuraba', 'Independencia', 
    'Isla de Maipo', 'La Cisterna', 'La Florida', 'La Granja', 'La Pintana', 'La Reina', 'Lampa', 
    'Las Condes', 'Lo Barnechea', 'Lo Espejo', 'Lo Prado', 'Macul', 'Maipú', 'María Pinto', 'Melipilla', 
    'Padre Hurtado', 'Paine', 'Pedro Aguirre Cerda', 'Peñaflor', 'Peñalolén', 'Pirque', 'Providencia', 
    'Pudahuel', 'Puente Alto', 'Quilicura', 'Quinta Normal', 'Recoleta', 'Renca', 'San Bernardo', 
    'San Joaquín', 'San José de Maipo', 'San Miguel', 'San Pedro', 'San Ramón', 'Santiago', 'Talagante', 
    'Tiltil', 'Vitacura', 'Ñuñoa'
]

#%%
# INICIO DE LA PÁGINA
st.set_page_config(page_title="Ambiental", layout='wide', initial_sidebar_state='expanded')

# Sidebar
st.sidebar.write("## Tablero Interactivo de Comunas: Indicadores priorizados")
comuna_seleccionada = st.sidebar.selectbox("Comuna:", lista_comunas, index=lista_comunas.index("Todas las comunas"))
current_year = datetime.now().year
select_year_int = st.sidebar.slider("Año:", min_value=2002, max_value=2035, value=current_year)
select_year = f'Poblacion {select_year_int}'

# TITULO INTRODUCCION
st.write('# Región Metropolitana y sus comunas: Ambiental')
st.write('Este tablero interactivo presenta indicadores ambientales priorizados de la Región Metropolitana de Santiago y sus comunas, proporcionando una visión detallada sobre las emisiones, el consumo de agua, y otros aspectos relevantes para la gestión ambiental y la salud pública.')


#%%

# Emisiones de dióxido de carbono para fuentes puntuales
st.write('## Emisiones de dióxido de carbono para fuentes puntuales')
st.write('Las emisiones de dióxido de carbono (CO₂) son un indicador clave para evaluar la contribución de diferentes fuentes a la contaminación del aire y al cambio climático. Este gráfico muestra las emisiones de CO₂ por fuentes puntuales de las regiones.')
infogram_url1 = "https://infogram.com/1prdpmyp12vd2xugvp1n53lre1smzy7prrw"
st.components.v1.iframe(infogram_url1, width=800, height=600, scrolling=True)
st.write('fuente: https://retc.mma.gob.cl/indicadores/emisiones-al-aire/')
# Emisiones de material particulado respirable fino para fuentes puntuales
st.write('## Emisiones de material particulado respirable fino para fuentes puntuales')
st.write('El material particulado respirable fino (PM2.5) es una de las principales preocupaciones ambientales debido a sus efectos adversos en la salud humana. Este gráfico muestra las emisiones de PM2.5 por fuentes puntuales en la Región Metropolitana.')
infogram_url2 = "https://infogram.com/1pyxvngg60xnvwt3m51k5g1wv5tyw2xll29"
st.components.v1.iframe(infogram_url2, width=800, height=600, scrolling=True)
st.write('fuente: https://retc.mma.gob.cl/indicadores/emisiones-al-aire/')
infogram_url3 = "https://infogram.com/1pyxvngg60xnvwt3m51k5g1wv5tyw2xll29"
st.components.v1.iframe(infogram_url3, width=800, height=600, scrolling=True)
st.write('fuente: https://retc.mma.gob.cl/indicadores/emisiones-al-aire/')
infogram_url4 = 'https://infogram.com/1p0lpr29w29en1fex16ylp3wr9fnd9r0l1l'
st.components.v1.iframe(infogram_url4, width=800, height=600, scrolling=True)
st.write('fuente: https://retc.mma.gob.cl/indicadores/emisiones-al-aire/')
# Incidencia de intoxicaciones agudas por plaguicidas (IAP)

import streamlit as st

# Título de la sección
st.write('## Mapas Open Data')
st.write('## Resumen de Declaraciones de Establecimientos Industriales y/o Municipales 2019')

# Descripción de la sección
st.write("""
Contiene el resumen de las declaraciones de los Establecimientos industriales y/o municipales registrados en Ventanilla Única durante el año 2019, de los siguientes sistemas:

- Sistema de Declaración y Seguimiento de Residuos Peligrosos (SIDREP)
- Declaración de Emisiones Atmosféricas (Formulario F138)
- Sistema Nacional de Declaración de Residuos (SINADER)

### Diccionario de datos:

- **Región**: Región donde se localiza el Establecimiento
- **Comuna**: Comuna donde se localiza el Establecimiento
- **Latitud**: Latitud del establecimiento
- **Longitud**: Longitud del establecimiento
- **Empresa**: Razón Social del Establecimiento
- **Establecim**: Nombre del Establecimiento
- **Rubro**: Rubro del Establecimiento
- **ID_VU**: Número Identificador del Establecimiento en RETC
- **CO2**: Emisiones de Dióxido de carbono (CO2) del Establecimiento durante el año 2019 (medida en tonelada/año)
- **MP**: Emisiones de Material Particulado durante el año 2019 (medida en tonelada/año)
- **SO2**: Emisiones de Dióxido de Azufre (SO2) del Establecimiento durante el año 2019 (medida en tonelada/año)
- **NOX**: Emisiones de Óxidos de nitrógeno (NOx) del Establecimiento durante el año 2019 (medida en tonelada/año)
- **RESPEL**: Generación de Residuos Peligrosos del Establecimiento durante el año 2019 (medida en tonelada/año)
- **RESNOPEL**: Generación de Residuos No Peligrosos del Establecimiento durante el año 2019 (medida en tonelada/año)
- **DESTINATAR**: Destinatario de Residuos No Peligrosos del Establecimiento durante el año 2019 (medida en tonelada/año)
""")

# Incrustar el mapa de OpenStreetMap usando un iframe
map_url = "https://umap.openstreetmap.fr/es/map/declaracion-jurada-anual-2018-toneladas_517111?scaleControl=true&miniMap=true&scrollWheelZoom=true&zoomControl=true&allowEdit=false&moreControl=false&searchControl=true&tilelayersControl=true&embedControl=null&datalayersControl=null&onLoadPanel=undefined&captionBar=false&fullscreenControl=true#6/-35.693/-72.378"
st.components.v1.iframe(map_url, width=800, height=600, scrolling=True)


st.write('## Incidencia de intoxicaciones agudas por plaguicidas (IAP)')
st.write('La incidencia de intoxicaciones agudas por plaguicidas es un importante indicador de los riesgos asociados al uso de estos productos químicos en la agricultura y otros sectores.')


st.write('## Riesgo de impactos de salud a consecuencias de olas de calor')
st.write('## Consumo promedio de agua por comuna')
st.write('## Decretos de Zonas de Escasez Hídrica 2008-2023')
st.write('## Nº de VIRS (instalaciones de interés sanitario) por comuna + MAPA')
# %%