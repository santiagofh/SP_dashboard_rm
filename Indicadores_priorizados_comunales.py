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
logo_url = "https://upload.wikimedia.org/wikipedia/commons/6/6e/SEREMISALUDMET.png"
st.image(logo_url, width=200) 
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


st.markdown('# Indicadores Censo 2017 y proyecciones')
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
# #%%
st.write("## Indicadores Socioeconómicos: Pobreza")
st.write(f"### Pobreza Multidimensional en {comuna_seleccionada}")
casen22_pobreza_m_comuna = casen22_pobreza_m[casen22_pobreza_m['Comuna'] == comuna_seleccionada]

# Calcular el total de la población
total_population = casen22_pobreza_m_comuna.iloc[0]['Pobres'] + casen22_pobreza_m_comuna.iloc[0]['No pobres']

# Calcular porcentajes
percentage_poor = (casen22_pobreza_m_comuna.iloc[0]['Pobres'] / total_population) * 100
percentage_not_poor = (casen22_pobreza_m_comuna.iloc[0]['No pobres'] / total_population) * 100

# Formatear los porcentajes con coma y dos decimales
formatted_percentage_poor = "{:,.2f}%".format(percentage_poor)
formatted_percentage_not_poor = "{:,.2f}%".format(percentage_not_poor)

# Mostrar métricas
cols = st.columns(2)
cols[0].metric('Pobres', formatted_percentage_poor)
cols[1].metric('No pobres', formatted_percentage_not_poor)

# Preparar datos para el gráfico de tarta
pie_data = {
    'Category': ['Pobres', 'No pobres'],
    'Values': [percentage_poor, percentage_not_poor]
}
df_pie = pd.DataFrame(pie_data)

# Visualización del gráfico de tarta
fig = px.pie(df_pie, values='Values', names='Category',
             title=f"Distribución de Pobreza Multidimensional en {comuna_seleccionada}")
st.plotly_chart(fig)

#%%
import pandas as pd
import plotly.express as px
import streamlit as st

st.write(f"### Pobreza Multidimensional en {comuna_seleccionada}")

# Supongamos que 'casen22_ingresos_comuna' es tu DataFrame con los datos de ingresos
# Filtrar datos por la comuna seleccionada
casen22_ingresos_comuna = casen22_ingresos[casen22_ingresos['Comuna'] == comuna_seleccionada]

# Formatear los valores de ingreso
casen22_ingresos_comuna['Ingresos del trabajo'] = casen22_ingresos_comuna['Ingresos del trabajo'].apply(lambda x: f"${x:,.0f}")
casen22_ingresos_comuna['Ingreso Autónomo'] = casen22_ingresos_comuna['Ingreso Autónomo'].apply(lambda x: f"${x:,.0f}")
casen22_ingresos_comuna['Ingreso Monetario'] = casen22_ingresos_comuna['Ingreso Monetario'].apply(lambda x: f"${x:,.0f}")
casen22_ingresos_comuna['Ingreso Total'] = casen22_ingresos_comuna['Ingreso Total'].apply(lambda x: f"${x:,.0f}")

# Crear una columna para categorías y una para valores en formato string
ingresos_melted = casen22_ingresos_comuna.melt(id_vars=['Comuna'], value_vars=['Ingresos del trabajo', 'Ingreso Autónomo', 'Ingreso Monetario', 'Ingreso Total'],
                                     var_name='Tipo de Ingreso', value_name='Valor')

# Crear gráfico de barras
fig = px.bar(ingresos_melted, x='Tipo de Ingreso', y='Valor', text='Valor',
             title=f"Ingresos en {comuna_seleccionada}", color='Tipo de Ingreso')

# Mostrar gráfico
st.plotly_chart(fig)

# Si deseas también puedes usar los datos para mostrar en métricas simples, ejemplo:
cols = st.columns(4)
cols[0].metric("Ingresos del trabajo", casen22_ingresos_comuna.iloc[0]['Ingresos del trabajo'])
cols[1].metric("Ingreso Autónomo", casen22_ingresos_comuna.iloc[0]['Ingreso Autónomo'])
cols[2].metric("Ingreso Monetario", casen22_ingresos_comuna.iloc[0]['Ingreso Monetario'])
cols[3].metric("Ingreso Total", casen22_ingresos_comuna.iloc[0]['Ingreso Total'])

# %%
import pandas as pd
import plotly.express as px
import streamlit as st

import pandas as pd
import plotly.express as px
import streamlit as st

st.write(f"### Participación laboral en {comuna_seleccionada}")

# Supongamos que 'casen22_participacion_lab' es tu DataFrame con los datos de participación laboral
# Filtrar datos por la comuna seleccionada
participacion_data = casen22_participacion_lab[casen22_participacion_lab['Comuna'] == comuna_seleccionada]

# Crear una columna para categorías y una para valores
participacion_melted = participacion_data.melt(id_vars=['Comuna'], value_vars=['Hombres', 'Mujeres', 'Total'],
                                     var_name='Grupo', value_name='Participación')

# Formatear los valores de participación como porcentajes con dos decimales y añadir el símbolo de porcentaje
participacion_melted['Participación'] = participacion_melted['Participación'].apply(lambda x: f"{x:.2f}%")

# Crear gráfico de barras para visualizar la participación
fig = px.bar(participacion_melted, x='Grupo', y='Participación', text='Participación',
             title=f"Participación Laboral en {comuna_seleccionada}", color='Grupo')

# Mostrar gráfico
st.plotly_chart(fig)

# Mostrar métricas utilizando los datos originales antes de la transformación y aplicando el formato
cols = st.columns(3)
cols[0].metric("Hombres", f"{participacion_data.iloc[0]['Hombres']:.2f}%")
cols[1].metric("Mujeres", f"{participacion_data.iloc[0]['Mujeres']:.2f}%")
cols[2].metric("Total", f"{participacion_data.iloc[0]['Total']:.2f}%")

#%%

import pandas as pd
import plotly.express as px
import streamlit as st

st.write(f"### Información sobre Migrantes en {comuna_seleccionada}")

# Suponemos que 'casen22_migrantes' es tu DataFrame con los datos sobre migrantes
# Filtrar datos por la comuna seleccionada
migrantes_data = casen22_migrantes[casen22_migrantes['Comuna'] == comuna_seleccionada]

# Crear una columna para categorías y una para valores
migrantes_melted = migrantes_data.melt(id_vars=['Comuna'], value_vars=['Ponlación nacida en Chile', 'Población nacida fuera de Chile', 'No sabe'],
                                      var_name='Categoría', value_name='Porcentaje')

# Crear gráfico de torta para visualizar los datos de migrantes
fig = px.pie(migrantes_melted, names='Categoría', values='Porcentaje', 
             title=f"Distribución de la Población por Origen en {comuna_seleccionada}",
             labels={'Porcentaje':'% del Total'})

# Mostrar gráfico
st.plotly_chart(fig)

# Opcional: Mostrar métricas utilizando los datos originales antes de la transformación y aplicando el formato
cols = st.columns(3)
cols[0].metric("Población nacida en Chile", f"{migrantes_data.iloc[0]['Ponlación nacida en Chile']:.2f}%")
cols[1].metric("Población nacida fuera de Chile", f"{migrantes_data.iloc[0]['Población nacida fuera de Chile']:.2f}%")
cols[2].metric("No sabe", f"{migrantes_data.iloc[0]['No sabe']:.2f}%")

#%%
import pandas as pd
import plotly.express as px
import streamlit as st

st.write(f"### Información sobre Pertenencia Étnica en {comuna_seleccionada}")

# Suponemos que 'casen22_etnias' es tu DataFrame con los datos sobre etnias
# Filtrar datos por la comuna seleccionada
etnias_data = casen22_etnias[casen22_etnias['Comuna'] == comuna_seleccionada]

# Crear una columna para categorías y una para valores
etnias_melted = etnias_data.melt(id_vars=['Comuna'], value_vars=['Pertenece a algún pueblo originario', 'No pertenece a ningún pueblo originario'],
                                var_name='Categoría', value_name='Porcentaje')

# Crear gráfico de torta para visualizar los datos étnicos
fig = px.pie(etnias_melted, names='Categoría', values='Porcentaje', 
             title=f"Distribución Étnica en {comuna_seleccionada}",
             labels={'Porcentaje':'% del Total'})

# Mostrar gráfico
st.plotly_chart(fig)

# Opcional: Mostrar métricas utilizando los datos originales antes de la transformación y aplicando el formato
cols = st.columns(2)
cols[0].metric("Pertenece a algún pueblo originario", f"{etnias_data.iloc[0]['Pertenece a algún pueblo originario']:.2f}%")
cols[1].metric("No pertenece a ningún pueblo originario", f"{etnias_data.iloc[0]['No pertenece a ningún pueblo originario']:.2f}%")

#%%
import pandas as pd
import plotly.express as px
import streamlit as st

st.write(f"### Información sobre Afiliación a Sistemas de Previsión en {comuna_seleccionada}")

# Suponemos que 'casen22_prevision' es tu DataFrame con los datos sobre afiliación a sistemas de previsión
# Filtrar datos por la comuna seleccionada
prevision_data = casen22_prevision[casen22_prevision['Comuna'] == comuna_seleccionada]

# Crear una columna para categorías y una para valores
prevision_melted = prevision_data.melt(id_vars=['Comuna'], value_vars=['fonasa', 'ff.aa. y del orden', 'isapre', 'ninguno (particular)', 'otro sistema', 'no sabe'],
                                      var_name='Tipo de Previsión', value_name='Porcentaje')

color_map = {
    'fonasa': '0068c9',
    'isapre': '83c9ff',
    'ff.aa. y del orden': '29b09d',
    'otro sistema': '7defa1',
    'no sabe': 'ffabab',
    'ninguno (particular)': 'ff2b2b',
}

# Crear gráfico de torta para visualizar la afiliación a sistemas de previsión
fig = px.pie(prevision_melted, names='Tipo de Previsión', values='Porcentaje', 
             title=f"Distribución de Afiliación a Sistemas de Previsión en {comuna_seleccionada}",
             labels={'Porcentaje':'% del Total'},
             color='Tipo de Previsión',
             color_discrete_map=color_map)  # Aplicar el mapa de colores

# Mostrar gráfico
st.plotly_chart(fig)

# Opcional: Mostrar métricas utilizando los datos originales antes de la transformación y aplicando el formato
cols = st.columns(6)
cols[0].metric("FONASA", f"{prevision_data.iloc[0]['fonasa']:.2f}%")
cols[1].metric("FF.AA. y del Orden", f"{prevision_data.iloc[0]['ff.aa. y del orden']:.2f}%")
cols[2].metric("ISAPRE", f"{prevision_data.iloc[0]['isapre']:.2f}%")
cols[3].metric("Ninguno (Particular)", f"{prevision_data.iloc[0]['ninguno (particular)']:.2f}%")
cols[4].metric("Otro Sistema", f"{prevision_data.iloc[0]['otro sistema']:.2f}%")
cols[5].metric("No Sabe", f"{prevision_data.iloc[0]['no sabe']:.2f}%")
#%%
#%%
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
