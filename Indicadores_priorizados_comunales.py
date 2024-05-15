#%% 
# Librerias a importar

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
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import statsmodels.api as sm
#%% 
# Path o rutas para archivos
path_censo17 = 'data_clean/CENSO17_Poblacion_rm.csv'
# https://www.ine.gob.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda/censo-de-poblacion-y-vivienda
path_geo='data_clean/Comunas_RM.geojson'
# https://www.ine.gob.cl/herramientas/portal-de-mapas/geodatos-abiertos
path_ine_proy='data_clean/INE_Proyecciones_RM.csv'
# https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion   
path_casen22_pobreza_m='data_clean/casen22_pobrezam.csv'
path_casen22_ingresos='data_clean/casen22_ingresos.csv'
path_casen22_participacion_lab='data_clean/casen22_participacion_laboral.csv'
path_casen22_migrantes='data_clean/casen22_migrantes.csv'
path_casen22_etnias='data_clean/casen22_etnias.csv'
path_casen22_prevision='data_clean/casen22_prevision.csv'
path_def='data_clean/Defunciones_2022_2024.csv'
path_casen='data_clean/casen_17_22.json'
#%% 
# LECTURA DE ARCHIVOS

ine17=pd.read_csv(path_ine_proy)
censo17 = pd.read_csv(path_censo17)
gdf = gpd.read_file(path_geo)
casen22_pobreza_m = pd.read_csv(path_casen22_pobreza_m)
casen22_ingresos = pd.read_csv(path_casen22_ingresos)
casen22_participacion_lab = pd.read_csv(path_casen22_participacion_lab)
casen22_migrantes = pd.read_csv(path_casen22_migrantes)
casen22_etnias = pd.read_csv(path_casen22_etnias)
casen22_prevision = pd.read_csv(path_casen22_prevision)
with open(path_casen, 'r') as json_file:
    json_data_dict = json.load(json_file)
casen_dict = {sheet: pd.DataFrame(data) for sheet, data in json_data_dict.items()}
for sheet_name, df in casen_dict.items():
    print(f"{sheet_name}")

defunciones=pd.read_csv(path_def)
#%%
# Listado comunas

lista_comunas=list(ine17['Nombre Comuna'].unique())
lista_comunas=sorted(lista_comunas)

#%% 
# INICIO DE LA PAGINA

st.set_page_config(page_title="Análisis de Comunas en Región Metropolitana", layout='wide', initial_sidebar_state='expanded')

#%%
# Sidebar 1
logo_url = "img/LOGO-MINSAL100-ANOS_color-original-1.png"

st.sidebar.image(logo_url, use_column_width=True)


st.sidebar.header("Selección de Comuna")
comunas = lista_comunas
default_index = comunas.index("Santiago") if "Santiago" in comunas else 0
comuna_seleccionada = st.sidebar.selectbox("Comuna:", comunas, index=default_index)

# Sidebar 2
st.sidebar.header("Selección año de proyección de población del INE")
current_year = datetime.now().year
select_year_int = st.sidebar.slider("Año:", min_value=2002, max_value=2035, value=current_year)
select_year = f'Poblacion {select_year_int}'
#%%
# TITULO INTRODUCCION

st.markdown('# Región Metropolitana y sus comunas: Indicadores priorizados')
# col1, col2 = st.columns([1, 3])
# with col1:

# with col2:
st.write("""
    ### Bienvenido al Tablero Interactivo de Comunas

    Este tablero está diseñado para proporcionar una visión integral y detallada de los diversos indicadores socioeconómicos, demográficos y urbanos de las comunas que forman la Región Metropolitana de Santiago. Aquí podrás explorar y visualizar datos que abarcan desde distribuciones de población y niveles de ingresos hasta aspectos de salud y educación.

    Utiliza los menús desplegables para seleccionar diferentes comunas y visualizar datos específicos relacionados con cada una. El tablero está enriquecido con gráficos interactivos que te ayudarán a entender mejor las características y desafíos de cada comuna.

    ### Cómo usar este tablero

    - **Selecciona una comuna:** Utiliza el menú desplegable para elegir una comuna y automáticamente se actualizarán los gráficos y tablas para reflejar los datos correspondientes.
    - **Explora los gráficos:** Interactúa con los gráficos para obtener detalles específicos sobre diferentes indicadores.
    - **Comparación y análisis:** Compara datos entre diferentes comunas ajustando tu selección y analiza las tendencias y patrones que emergen de los datos.
    """)

# %%
# Filtro de comuna seleccionada
gdf_comuna=gdf_comuna = gdf[gdf['NOM_COMUNA'] == str.upper(comuna_seleccionada)]
ine17_comuna=ine17.loc[ine17['Nombre Comuna']==comuna_seleccionada]
censo17_comuna=censo17.loc[censo17['NOMBRE COMUNA']==str.upper(comuna_seleccionada)]

#%% 
# Calculo de pop y densidad
pop_proy_total=ine17_comuna[select_year].sum()
area_comuna=1
densidad_pop=pop_proy_total/area_comuna

# %%
# 
# Load and prepare data
current_year = datetime.now().year
year_column = f'Poblacion {current_year}'

# Assuming censo17_comuna and ine17_comuna are already loaded with appropriate data
censo17_comuna_pop = censo17_comuna.loc[censo17_comuna.EDAD == 'Total Comuna']
censo17_comuna_edad = censo17_comuna.loc[censo17_comuna.EDAD != 'Total Comuna']

# Census data
pop_censada = censo17_comuna_pop['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA']
pop_h = censo17_comuna_pop['HOMBRES ']
pop_m = censo17_comuna_pop['MUJERES']
pop_h_percentaje = (pop_h / pop_censada * 100).iloc[0]
pop_m_percentaje = (pop_m / pop_censada * 100).iloc[0]
pop_urb_percentage = (censo17_comuna_pop['TOTAL ÁREA URBANA'] / censo17_comuna_pop['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA'] * 100).iloc[0]
pop_rur_percentage = (censo17_comuna_pop['TOTAL ÁREA RURAL'] / censo17_comuna_pop['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA'] * 100).iloc[0]


# Projected data for 2024
pop_proy_total = ine17_comuna[select_year].sum()
pop_proy_h = ine17_comuna.loc[ine17_comuna['Sexo (1=Hombre 2=Mujer)'] == 1, select_year].sum()
pop_proy_m = ine17_comuna.loc[ine17_comuna['Sexo (1=Hombre 2=Mujer)'] == 2, select_year].sum()
pop_proy_h_percentaje=pop_proy_h/pop_proy_total*100
pop_proy_m_percentaje=pop_proy_m/pop_proy_total*100
# Current year gender distribution
pop_current_year = ine17_comuna[['Sexo (1=Hombre 2=Mujer)', year_column]].dropna()
gender_population = pop_current_year.groupby('Sexo (1=Hombre 2=Mujer)')[year_column].sum().reset_index()
total_population = gender_population[year_column].sum()
gender_population['Percentage'] = (gender_population[year_column] / total_population) * 100
gender_population['Formatted_Percentage'] = gender_population['Percentage'].apply(lambda x: "{:.2f}%".format(x))
fig = px.pie(gender_population, values='Percentage', names='Sexo (1=Hombre 2=Mujer)',
             title="Distribución de Género 2024")


# Utilizando comprensión de lista y función map para aplicar el formateo y reemplazo en una sola línea si es necesario.
formatted_values = {
    "pop_censada": f"{int(pop_censada):,}".replace(',', '.'),
    "pop_h": f"{int(pop_h):,}".replace(',', '.'),
    "pop_m": f"{int(pop_m):,}".replace(',', '.'),
    "pop_h_percentaje": "{:.2f}%".format(pop_h_percentaje),
    "pop_m_percentaje": "{:.2f}%".format(pop_m_percentaje),
    "pop_urb_percentage": f"{pop_urb_percentage:.2f}%",
    "pop_rur_percentage": f"{pop_rur_percentage:.2f}%",
    "pop_proy_total": f"{int(pop_proy_total):,}".replace(',', '.'),
    "pop_proy_h": f"{int(pop_proy_h):,}".replace(',', '.'),
    "pop_proy_m": f"{int(pop_proy_m):,}".replace(',', '.'),
    'pop_proy_h_percentaje':"{:.2f}%".format(pop_proy_h_percentaje),
    'pop_proy_m_percentaje':"{:.2f}%".format(pop_proy_m_percentaje),
}

# Ahora puedes acceder a cada valor formateado desde el diccionario
formatted_pop_censada = formatted_values["pop_censada"]
formatted_pop_h = formatted_values["pop_h"]
formatted_pop_m = formatted_values["pop_m"]
formatted_pop_h_percentaje = formatted_values["pop_h_percentaje"]
formatted_pop_m_percentaje = formatted_values["pop_m_percentaje"]
formatted_pop_urb = formatted_values["pop_urb_percentage"]
formatted_pop_rur = formatted_values["pop_rur_percentage"]
formatted_pop_proy_total = formatted_values["pop_proy_total"]
formatted_pop_proy_h = formatted_values["pop_proy_h"]
formatted_pop_proy_m = formatted_values["pop_proy_m"]
formatted_pop_proy_h_percent = formatted_values["pop_proy_h_percentaje"]
formatted_pop_proy_m_percent = formatted_values["pop_proy_m_percentaje"]


st.markdown('## Indicadores territoriales y de población')
cols = st.columns(5)

cols[0].metric("Población efectivamente censada 2017", formatted_pop_censada)
cols[1].metric("Total hombres (censo 2017)", formatted_pop_h)
cols[2].metric("Total mujeres (censo 2017)", formatted_pop_m)
cols[3].metric("Porcentaje de hombres (censo2017)", formatted_pop_h_percentaje)
cols[4].metric("Porcentaje de mujeres (censo2017)", formatted_pop_m_percentaje)


cols[0].metric(f"Población proyectada de la comuna ({select_year})", formatted_pop_proy_total)
cols[1].metric(f"Total hombres ({select_year})", formatted_pop_proy_h)
cols[2].metric(f"Total mujeres ({select_year})", formatted_pop_proy_m)
cols[3].metric(f"Porcentaje de hombres ({select_year})",formatted_pop_proy_h_percent)
cols[4].metric(f"Porcentaje de mujeres ({select_year})",formatted_pop_proy_m_percent)


st.write('_Fuente: Elaboración propia a partir de INE 2017_')
st.write('_https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion_')
#%%
# Mapa de la comuna
def get_zoom_level(area):
    scaled_area = area * 1000 
    if scaled_area > 50:
        return 10
    elif scaled_area > 20:
        return 11
    elif scaled_area > 10:
        return 11
    elif scaled_area > 5:
        return 12
    else:
        return 13
st.write(f"## Visualizar mapa de la comuna {comuna_seleccionada}")

if not gdf_comuna.empty:
    area = gdf_comuna.geometry.area.iloc[0]  # Get the area of the comuna
    centroid = gdf_comuna.geometry.centroid.iloc[0]
    zoom_start = get_zoom_level(area)  # Dynamic zoom level based on area
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=zoom_start)
    # st.write(area*1000)
    folium.GeoJson(gdf_comuna, name='geojson').add_to(m)
    folium.LayerControl().add_to(m)
    folium_static(m)
else:
    st.write("No se encontró la comuna seleccionada en los datos geográficos.")
area_comuna=gdf_comuna.Superf_KM2
pop_comuna = ine17_comuna[select_year].sum()
densidad_pop=pop_comuna/area_comuna

cols = st.columns(4)
cols[0].metric("Área total de la comuna (población proyectada 2024)", f"{int(area_comuna)} km²")
cols[1].metric("Densidad poblacional de la comuna (población proyectada)", f"{int(densidad_pop)} hab/km²")
cols[0].metric("Porcentaje área urbana (censo 2017)", formatted_pop_urb)
cols[1].metric("Porcentaje área rural (censo 2017)", formatted_pop_rur)

st.write('_Fuente: Elaboración propia a partir de datos geograficos nacionales (https://www.ine.gob.cl/herramientas/portal-de-mapas/geodatos-abiertos)_')
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
fig.add_vline(x=select_year_int, line_width=2, line_dash="dash", line_color="red")

fig.add_annotation(
    x=select_year_int, 
    # y=0.95,  # ajusta esta ubicación según la escala de tu gráfico
    xref="x",
    yref="paper",
    text=f"Año: {select_year_int}",
    showarrow=False,
    font=dict(color="red", size=15)
)


fig.update_layout(
    yaxis_title='Total población',
    xaxis_title='Año',
    legend_title='Sexo',

)
st.plotly_chart(fig)
st.write('_Fuente: Elaboración propia a partir de INE 2017_')
st.write('_https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion_')
# %%
st.write(f'## Piramide poblacional para {comuna_seleccionada}')



ine17_comuna['Sexo'] = ine17_comuna['Sexo (1=Hombre 2=Mujer)'].map({1: 'Hombres', 2: 'Mujeres'})
years = [f'Poblacion {year}' for year in range(2002, 2036)]
data_melted = ine17_comuna.melt(id_vars=['Edad', 'Sexo'], value_vars=years, var_name='Year', value_name='Population')
data_melted['Year'] = data_melted['Year'].str.extract('(\d+)').astype(int)
grouped_data = data_melted.groupby(['Year', 'Sexo', 'Edad']).agg({'Population': 'sum'}).reset_index()
def age_group(age):
    if age >= 80:
        return "80 y más"
    elif age >= 75:
        return "75 a 79"
    elif age >= 70:
        return "70 a 74"
    elif age >= 65:
        return "65 a 69"
    elif age >= 60:
        return "60 a 64"
    elif age >= 55:
        return "55 a 59"
    elif age >= 50:
        return "50 a 54"
    elif age >= 45:
        return "45 a 49"
    elif age >= 40:
        return "40 a 44"
    elif age >= 35:
        return "35 a 39"
    elif age >= 30:
        return "30 a 34"
    elif age >= 25:
        return "25 a 29"
    elif age >= 20:
        return "20 a 24"
    elif age >= 15:
        return "15 a 19"
    elif age >= 10:
        return "10 a 14"
    elif age >= 5:
        return "05 a 09"
    else:
        return "0 a 04"
grouped_data['Age Group'] = grouped_data['Edad'].apply(age_group)
grouped_data = grouped_data.groupby(['Year', 'Sexo', 'Age Group']).agg({'Population': 'sum'}).reset_index()
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=['Hombres', 'Mujeres'],
                    horizontal_spacing=0.02, x_title='Población')
for year in range(2002, 2036):
    for sexo in ['Hombres', 'Mujeres']:
        subset = grouped_data[(grouped_data['Year'] == year) & (grouped_data['Sexo'] == sexo)]
        subset = subset.sort_values(by='Age Group')
        fig.add_trace(
            go.Bar(x=-subset['Population'] if sexo == 'Hombres' else subset['Population'], y=subset['Age Group'],
                   orientation='h', name=sexo, visible=(year == datetime.now().year)),
            1, 1 if sexo == 'Hombres' else 2
        )
steps = []
for i, year in enumerate(range(2002, 2036)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)}],
        label=str(year)
    )
    for j in range(len(fig.data)):
        step["args"][0]["visible"][2 * i + j % 2] = True
    steps.append(step)
year_index = select_year_int - 2002

sliders = [dict(
    active=year_index,
    currentvalue={"prefix": "Año: "},
    pad={"t": 50},
    steps=steps
)]

max_population = grouped_data['Population'].max()

# Debemos asegurarnos de usar este máximo para ambos lados del eje X
fig.update_layout(
    sliders=sliders,
    title=f"Pirámide Poblacional de {comuna_seleccionada} por Año",
    xaxis_title="Población",
    yaxis_title="Rango Etario",
    showlegend=False,
    xaxis=dict(range=[-max_population, 0]),  # Ajusta el rango para ser simétrico
    xaxis2=dict(range=[0, max_population])  # Asegúrate de hacer esto para ambos ejes X si estás usando subplots
)


st.plotly_chart(fig)
st.write('_Fuente: Elaboración propia a partir de INE 2017_')
st.write('_https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion_')
#%%
st.write("## Indicadores Socioeconómicos")
#%%
st.write(f"### Pobreza de ingresos para {comuna_seleccionada}")
casen_pobrezai = casen_dict['POBREZA DE INGRESOS']
casen_pobrezai_comuna = casen_pobrezai[casen_pobrezai['Comuna'] == comuna_seleccionada]
casen_pobrezai_comuna['Pobres'].fillna(casen_pobrezai_comuna['Pobres 2020'], inplace=True)
casen_pobrezai_comuna['Pobres'] = casen_pobrezai_comuna['Pobres'] / 100
fig = px.bar(
    casen_pobrezai_comuna,
    x='Año',
    y='Pobres',
    title=f"Pobreza de ingresos en {comuna_seleccionada}",
    labels={'Pobres': 'Porcentaje de Pobreza de Ingresos', 'Año': 'Año'}
)
fig.update_layout(
    yaxis_tickformat=",.2%"
)
st.plotly_chart(fig)
#%%
st.write(f"### Pobreza multidimencional para {comuna_seleccionada}")
casen_pobrezam = casen_dict['POBREZA MULTIDIMENSIONAL']
casen_pobrezam_comuna = casen_pobrezam[casen_pobrezam['Comuna'] == comuna_seleccionada]
casen_pobrezam_comuna['Pobres'] = casen_pobrezam_comuna['Pobres'] / 100
fig = px.bar(
    casen_pobrezam_comuna,
    x='Año',
    y='Pobres',
    title=f"Pobreza de ingresos en {comuna_seleccionada}",
    labels={'Pobres': 'Porcentaje de Pobreza Multidimensional', 'Año': 'Año'}
)
fig.update_layout(
    yaxis_tickformat=",.2%"
)
st.plotly_chart(fig)

#%%
st.write(f"### Ingresos en {comuna_seleccionada}")
casen_ingresos = casen_dict['INGRESOS']
casen_ingresos_comuna = casen_ingresos[casen_ingresos['Comuna'] == comuna_seleccionada]
casen_ingresos_comuna_long = pd.melt(
    casen_ingresos_comuna,
    id_vars=['Año'],
    value_vars=['Ingreso Autónomo', 'Ingreso Monetario', 'Ingresos del trabajo', 'Ingreso Total'],
    var_name='Tipo de Ingreso',
    value_name='Monto'
)
fig = px.bar(
    casen_ingresos_comuna_long,
    x='Año',
    y='Monto',
    color='Tipo de Ingreso',
    barmode='group',
    title=f"Distribución de Ingresos en {comuna_seleccionada}",
    labels={'Monto': 'Monto de Ingreso', 'Año': 'Año'}
)

fig.update_layout(
    yaxis_tickformat=".0f"
)
st.plotly_chart(fig)
# %%
st.write(f"### Participación laboral en {comuna_seleccionada}")

casen_tasas_participacion = casen_dict['TASAS PARTICIPACIÓN LABORAL']
casen_tasas_participacion_comuna = casen_tasas_participacion[casen_tasas_participacion['Comuna'] == comuna_seleccionada]
casen_tasas_participacion_comuna['Hombres']=casen_tasas_participacion_comuna['Hombres']/100
casen_tasas_participacion_comuna['Mujeres']=casen_tasas_participacion_comuna['Mujeres']/100
# Crear un DataFrame en formato largo para el gráfico
casen_tasas_participacion_comuna_long = pd.melt(
    casen_tasas_participacion_comuna,
    id_vars=['Año'],
    value_vars=['Hombres', 'Mujeres'],
    var_name='Sexo',
    value_name='Porcentaje'
)

# Crear el gráfico
fig = px.bar(
    casen_tasas_participacion_comuna_long,
    x='Año',
    y='Porcentaje',
    color='Sexo',
    barmode='group',
    title=f"Tasas de participación laboral en {comuna_seleccionada}",
    labels={'Porcentaje': 'Porcentaje de Participación', 'Año': 'Año'}
)

# Formatear las etiquetas del eje y como porcentajes
fig.update_layout(
    yaxis_tickformat=".2%",  # Formato de porcentaje con dos decimales
    yaxis_title='Porcentaje de Participación'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

#%%
st.write(f'### Migrantes en {comuna_seleccionada}')
casen_migrantes = casen_dict['MIGRANTES']
casen_migrantes_comuna = casen_migrantes[casen_migrantes['Comuna'] == comuna_seleccionada]
casen_migrantes_comuna['Población nacida en Chile']=casen_migrantes_comuna['Ponlación nacida en Chile']/100
casen_migrantes_comuna['Población nacida fuera de Chile']=casen_migrantes_comuna['Población nacida fuera de Chile']/100
casen_migrantes_comuna['No sabe']=casen_migrantes_comuna['No sabe']/100

# Crear un DataFrame en formato largo para el gráfico
casen_migrantes_comuna_long = pd.melt(
    casen_migrantes_comuna,
    id_vars=['Año'],
    value_vars=['Población nacida en Chile', 'Población nacida fuera de Chile', 'No sabe'],
    var_name='Tipo de Población',
    value_name='Porcentaje'
)

# Crear el gráfico de barras
fig_bar = px.bar(
    casen_migrantes_comuna_long,
    x='Año',
    y='Porcentaje',
    color='Tipo de Población',
    barmode='group',
    title=f"Distribución de la Población Migrante en {comuna_seleccionada}",
    labels={'Porcentaje': 'Porcentaje de la Población', 'Año': 'Año'}
)

# Crear el gráfico de dispersión con línea de tendencia
fig_trend = px.scatter(
    casen_migrantes_comuna_long[casen_migrantes_comuna_long['Tipo de Población'] == 'Población nacida fuera de Chile'],
    x='Año',
    y='Porcentaje',
    trendline="ols",
    labels={'Porcentaje': 'Porcentaje de la Población', 'Año': 'Año'}
)

# Combinar los gráficos
fig_trend.data[1].marker.color = 'red'
fig_trend.data[1].name = 'Tendencia de Migrantes'
fig_trend.data[0].showlegend = False
fig = fig_bar.add_traces(fig_trend.data[1:])

# Formatear las etiquetas del eje y
fig.update_layout(
    yaxis_tickformat=".2%",  # Formato de porcentaje con dos decimales
    yaxis_title='Porcentaje de la Población'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)




#%%

st.write(f"### Pertenencia Étnica en {comuna_seleccionada}")

casen_etnias = casen_dict['ETNIAS']
casen_etnias_comuna = casen_etnias[casen_etnias['Comuna'] == comuna_seleccionada]

# Crear un DataFrame en formato largo para el gráfico
casen_etnias_comuna_long = pd.melt(
    casen_etnias_comuna,
    id_vars=['Año'],
    value_vars=['Pertenece a algún pueblo originario', 'No pertenece a ningún pueblo originario'],
    var_name='Tipo de Población',
    value_name='Porcentaje'
)

# Crear el gráfico
fig = px.bar(
    casen_etnias_comuna_long,
    x='Año',
    y='Porcentaje',
    color='Tipo de Población',
    barmode='group',
    title=f"Distribución de la población por pertenencia a pueblo originario en {comuna_seleccionada}",
    labels={'Porcentaje': 'Porcentaje de la Población', 'Año': 'Año'}
)

# Formatear las etiquetas del eje y como porcentajes
fig.update_layout(
    yaxis_tickformat=".2%",  # Formato de porcentaje con dos decimales
    yaxis_title='Porcentaje de la Población'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Mostrar los datos
st.write(casen_etnias_comuna)

#%%

st.write(f"### Afiliación a Sistemas de Previsión en {comuna_seleccionada}")

# Map of column renaming
column_rename_map = {
    'fonasa': 'FONASA',
    'ff.aa. y del orden': 'FF.AA. y del Orden',
    'isapre': 'Isapre',
    'ninguno (particular)': 'Ninguno (Particular)',
    'otro sistema': 'Otro Sistema',
    'no sabe': 'No Sabe'
}

# Color map for different types of health insurance
color_map = {
    'FONASA': '#0068c9',
    'Isapre': '#83c9ff',
    'FF.AA. y del Orden': '#29b09d',
    'Otro Sistema': '#7defa1',
    'No Sabe': '#ffabab',
    'Ninguno (Particular)': '#ff2b2b'
}

# Filtrar los datos de la pestaña "PREVISIÓN DE SALUD" para la comuna seleccionada
casen_prevision_salud = casen_dict['PREVISIÓN DE SALUD']
casen_prevision_salud_comuna = casen_prevision_salud[casen_prevision_salud['Comuna'] == comuna_seleccionada]

# Renombrar las columnas
casen_prevision_salud_comuna = casen_prevision_salud_comuna.rename(columns=column_rename_map)

# Dividir los valores por 100 para obtener porcentajes
for col in column_rename_map.values():
    if col in casen_prevision_salud_comuna.columns:
        casen_prevision_salud_comuna[col] = casen_prevision_salud_comuna[col] / 100

# Crear un DataFrame en formato largo para el gráfico
casen_prevision_salud_comuna_long = pd.melt(
    casen_prevision_salud_comuna,
    id_vars=['Año'],
    value_vars=list(column_rename_map.values()),
    var_name='Tipo de Previsión',
    value_name='Porcentaje'
)

# Crear el gráfico de barras
fig_bar = px.bar(
    casen_prevision_salud_comuna_long,
    x='Año',
    y='Porcentaje',
    color='Tipo de Previsión',
    barmode='group',
    title=f"Distribución de Previsión de Salud en {comuna_seleccionada}",
    labels={'Porcentaje': 'Porcentaje de la Población', 'Año': 'Año'},
    color_discrete_map=color_map
)

# Crear el gráfico de dispersión con línea de tendencia para FONASA
fig_trend = px.scatter(
    casen_prevision_salud_comuna,
    x='Año',
    y='FONASA',
    trendline="ols",
    labels={'FONASA': 'Porcentaje de la Población', 'Año': 'Año'}
)

# Combinar los gráficos
fig_trend.data[1].marker.color = 'red'
fig_trend.data[1].name = 'Tendencia FONASA'
fig_trend.data[0].showlegend = False
fig = fig_bar.add_traces(fig_trend.data[1:])

# Formatear las etiquetas del eje y como porcentajes
fig.update_layout(
    yaxis_tickformat=".2%",  # Formato de porcentaje con dos decimales
    yaxis_title='Porcentaje de la Población'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Mostrar los datos
st.write(casen_prevision_salud_comuna)
#%%
#%%
st.write(f"## Indicadores de defunciones")

defunciones['fecha_def'] = pd.to_datetime(defunciones['fecha_def'])
anio_actual = datetime.now().year
lista_anios = sorted(defunciones['fecha_def'].dt.year.unique(), reverse=True)
anio_seleccionado = st.selectbox("Seleccione el año:", lista_anios, index=lista_anios.index(anio_actual) if anio_actual in lista_anios else 0)
columna_poblacion = f'Poblacion {anio_seleccionado}'
poblacion_total = ine17_comuna[columna_poblacion].sum()

st.write(f"### Defunciones en {comuna_seleccionada} para el año {anio_seleccionado}")

poblacion_total_formatted=int(poblacion_total)
st.write(f"Población total estimada en {comuna_seleccionada} para el año {anio_seleccionado}: {poblacion_total_formatted}")

#%%
defunciones_comuna = defunciones[(defunciones['comuna'] == comuna_seleccionada) & (defunciones['fecha_def'].dt.year == anio_seleccionado)]
defunciones_comuna.rename(columns={'casusa_def': 'Causas de defunciones'}, inplace=True)
top_causas_def = defunciones_comuna['Causas de defunciones'].value_counts()
# Cambio aquí para mostrar una tabla en lugar de un gráfico
# st.table(top_causas_def.reset_index().rename(columns={'casusa_def': 'Causa de Defunciones'}))
st.dataframe(top_causas_def, height=300)
# %%
total_defunciones = defunciones_comuna.shape[0]
porcentaje_defunciones = (total_defunciones / poblacion_total) * 100
porcentaje_poblacion_viva = 100 - porcentaje_defunciones
data_pie = {
    'Categoría': ['Defunciones', 'Población Viva'],
    'Porcentaje': [porcentaje_defunciones, porcentaje_poblacion_viva]
}
#%%
import pandas as pd
import plotly.express as px
import streamlit as st

# Suponemos que 'defunciones' y 'ine17_comuna' son tus DataFrames ya cargados y listos para usar
ultimos_tres_anios = sorted(defunciones['fecha_def'].dt.year.unique(), reverse=True)[:3]

resultados = []
for anio in ultimos_tres_anios:
    defunciones_anio = defunciones[(defunciones['comuna'] == comuna_seleccionada) & (defunciones['fecha_def'].dt.year == anio)]
    poblacion_anio = ine17_comuna[f'Poblacion {anio}'].sum()
    total_defunciones_anio = defunciones_anio.shape[0]
    porcentaje_defunciones = (total_defunciones_anio / poblacion_anio) * 100
    resultados.append({
        'Año': str(anio),  # Convertir año a cadena para asegurar que Plotly lo trate como categórico
        'Porcentaje de Defunciones': porcentaje_defunciones,
        'Población Total': poblacion_anio
    })

df_resultados = pd.DataFrame(resultados)

fig_bar = px.bar(
    df_resultados, 
    x='Año', 
    y='Porcentaje de Defunciones', 
    title=f'Porcentaje de Defunciones en {comuna_seleccionada} para los últimos tres años',
    labels={'Porcentaje de Defunciones': '% de Defunciones'},
    hover_data=['Población Total'],
    category_orders={"Año": [str(y) for y in sorted(ultimos_tres_anios, reverse=False)]}  # Asegura el orden correcto y tratamiento categórico
)

fig_bar.update_xaxes(type='category')  # Esto hace que el eje X sea tratado como categórico
st.plotly_chart(fig_bar)
