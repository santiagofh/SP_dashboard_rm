#%% 
# LIBRERIAS

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import plotly.express as px
from datetime import datetime

#%% 
# RUTAS DE ARCHIVOS

path_censo17 = 'data_clean/CENSO17_Poblacion_rm.csv'
#https://www.ine.gob.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda/censo-de-poblacion-y-vivienda
path_geo='data_clean/Poligonos_comunas_RM.geojson'
#
path_ine_proy='data_clean/INE_Proyecciones_RM.csv'
# https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion   

#%% 
# LECTURA DE ARCHIVOS

ine17=pd.read_csv(path_ine_proy)
censo17 = pd.read_csv(path_censo17)
gdf = gpd.read_file(path_geo)
#%%
# Listado comunas

lista_comunas=list(ine17['Nombre Comuna'].unique())
lista_comunas=sorted(lista_comunas)

#%% 
# INICIO DE LA PAGINA

st.set_page_config(page_title="Análisis de Comunas en Región Metropolitana", layout='wide', initial_sidebar_state='expanded')

#%%
# Sidebar 1
st.sidebar.header("Selección de Comuna")
comunas = lista_comunas
default_index = comunas.index("Santiago") if "Santiago" in comunas else 0
comuna_seleccionada = st.sidebar.selectbox("Comuna:", comunas, index=default_index)

# Sidebar 2
st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Seleccione un tópico:",
    ('Indicadores territoriales', 'Pirámide Poblacional', 'Socioeconómico', 
     'Mortalidad General', 'Mortalidad GG', 'Mortalidad CE', 'Estilos Nutricional', 
     'Estilos de vida y FR', 'Estilos de vida y FR (2)', 'PNI', 'Salud', 'Mortalidad Infantil',
     'Pobreza', 'Diarréa', 'Egresos Hospitalarios')
)

#%%
# TITULO INTRODUCCION

st.markdown('# Región Metropolitana y sus comunas: Indicadores priorizados')
col1, col2 = st.columns([1, 3])
with col1:
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/6/6e/SEREMISALUDMET.png"
    st.image(logo_url) 
with col2:
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

# %%
# Filtro de comuna seleccionada
gdf_comuna=gdf_comuna = gdf[gdf['Comuna'] == comuna_seleccionada]
ine17_comuna=ine17.loc[ine17['Nombre Comuna']==comuna_seleccionada]
censo17_comuna=censo17.loc[censo17['NOMBRE COMUNA']==str.upper(comuna_seleccionada)]

#%% 
# Calculo de pop y densidad
pop_total_comuna=ine17_comuna['Poblacion 2023'].sum()
area_comuna=1
densidad_pop=pop_total_comuna/area_comuna

# %%
# Indicadores territoriales
st.markdown(f'# Indicadores territoriales')
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Población proyectada de la comuna 2024", pop_total_comuna)
col2.metric("Area total de la comuna", area_comuna)
col3.metric("Densidad poblacional de la comuna", densidad_pop)
st.write('_Fuente: Elaboración propia a partir de INE 2017_')

#%%
# Mapa de la comuna
st.write((f"## Visualizar mapa de la comuna {comuna_seleccionada}"))
if not gdf_comuna.empty:
    centroid = gdf_comuna.geometry.centroid.iloc[0]
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)
    folium.GeoJson(
        gdf_comuna,
        name='geojson'
    ).add_to(m)
    folium.LayerControl().add_to(m)
    folium_static(m)
else:
    st.write("No se encontró la comuna seleccionada en los datos geográficos.")
st.write('_Fuente: Elaboración propia a partir de datos geograficos nacionales_')
#%%
# Población proyectada
st.write('## Poblacion proyectada')
population_data = ine17_comuna[['Nombre Comuna', 'Sexo (1=Hombre 2=Mujer)'] + [f'Poblacion {year}' for year in range(2002, 2036)]]
population_data_melted = population_data.melt(id_vars=['Nombre Comuna', 'Sexo (1=Hombre 2=Mujer)'], var_name='Año', value_name='Población')
population_data_melted['Año'] = population_data_melted['Año'].str.extract('(\d+)').astype(int)
total_population_by_gender = population_data_melted.groupby(['Año', 'Sexo (1=Hombre 2=Mujer)'])['Población'].sum().reset_index()
total_population_by_gender.sort_values(by=['Año', 'Sexo (1=Hombre 2=Mujer)'], inplace=True)
total_population_by_gender['Sexo (1=Hombre 2=Mujer)'] = total_population_by_gender['Sexo (1=Hombre 2=Mujer)'].map({1: 'Hombre', 2: 'Mujer'})
fig = px.line(
    total_population_by_gender,
    x='Año',
    y='Población',
    color='Sexo (1=Hombre 2=Mujer)',
    title='Total población proyectada ' + comuna_seleccionada,
    labels={'Sexo (1=Hombre 2=Mujer)': 'Sexo', 'Año': 'Año', 'Población': 'Población'}
)
current_year = datetime.now().year
fig.add_vline(x=current_year, line_width=2, line_dash="dash", line_color="red")

fig.add_annotation(
    x=current_year, 
    # y=0.95,  # ajusta esta ubicación según la escala de tu gráfico
    xref="x",
    yref="paper",
    text=f"Año actual: {current_year}",
    showarrow=False,
    font=dict(color="red", size=14)
)


fig.update_layout(
    yaxis_title='Total población',
    xaxis_title='Año',
    legend_title='Sexo',

)
st.plotly_chart(fig)
st.write('_Fuente: Elaboración propia a partir de INE 2017_')
st.write('_https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion_')
#%%
st.markdown(f'# Indicadores Censo 2017')
censo17_comuna_pop=censo17_comuna.loc[censo17_comuna.EDAD=='Total Comuna']
censo17_comuna_edad=censo17_comuna.loc[censo17_comuna.EDAD!='Total Comuna']
pop_censada=censo17_comuna_pop['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA']
pop_h=censo17_comuna_pop['HOMBRES ']
pop_m=censo17_comuna_pop['MUJERES']
pop_urb=censo17_comuna_pop['TOTAL ÁREA URBANA']
pop_rur=censo17_comuna_pop['TOTAL ÁREA RURAL']
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
col1.metric("Población efectivamente censada 2017", pop_censada)
col2.metric("Total hombres", pop_h)
col3.metric("Total mujeres", pop_m)
col4.metric("Total área urbana", pop_urb)
col5.metric("Total área rural", pop_rur)
st.write('_Fuente: Elaboración propia a partir de CENSO 2017_ _(https://www.ine.gob.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda/censo-de-poblacion-y-vivienda)_')
# %%
st.markdown(f'# Piramide poblacional')
import pandas as pd
import plotly.graph_objects as go

# Suponiendo que censo17_comuna_edad ya está cargado como DataFrame
# Corregir el nombre de la columna "HOMBRE " a "HOMBRE"
censo17_comuna_edad.rename(columns={'HOMBRES ': 'HOMBRES'}, inplace=True)

# Eliminar columnas innecesarias si es necesario
columns_to_keep = ['EDAD', 'HOMBRES', 'MUJERES']
censo17_comuna_edad = censo17_comuna_edad[columns_to_keep]

# Convertir la columna de edad para manejar "100 o más" como un solo grupo
censo17_comuna_edad['EDAD'] = censo17_comuna_edad['EDAD'].replace('100 o más', '100+')

# Preparar los datos para la pirámide poblacional
censo17_comuna_edad.sort_values('EDAD', inplace=True)  # Asegurarse de que las edades están en orden ascendente

# Crear el gráfico
fig = go.Figure()

# Añadir la serie de datos para hombres
fig.add_trace(go.Bar(
    y=censo17_comuna_edad['EDAD'],
    x=-censo17_comuna_edad['HOMBRES'],  # Negativo para que apunte hacia la izquierda
    name='Hombres',
    orientation='h',
    marker=dict(color='blue')
))

# Añadir la serie de datos para mujeres
fig.add_trace(go.Bar(
    y=censo17_comuna_edad['EDAD'],
    x=censo17_comuna_edad['MUJERES'],
    name='Mujeres',
    orientation='h',
    marker=dict(color='red')
))

# Actualizar el layout del gráfico
fig.update_layout(
    title=f'Pirámide Poblacional de {comuna_seleccionada}',
    xaxis_title='Número de personas',
    yaxis_title='Edad',
    barmode='relative',
    bargap=0.1,  # Espacio entre las barras del gráfico
    bargroupgap=0.1  # Espacio entre grupos de barras
)

# Mostrar el gráfico
st.write(fig)


# %%
