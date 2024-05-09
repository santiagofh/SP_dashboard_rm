#%%
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Cargar datos
path_censo17 = 'data/1_2_POBLACION.xlsx'
path_casen22 = 'data/INDICADORES COMUNALES CASEN 2022 RMS.xlsx'
censo17 = pd.read_excel(path_censo17, sheet_name='Comuna', skiprows=2)
def read_casen(pestaña):
    df=pd.read_excel(path_casen22, sheet_name=pestaña, skiprows=4)
    df=df.replace('**',np.nan)
    df=df.replace('*',np.nan)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True) 
    return df

#%%
casen22_pobrezam=read_casen('POBREZA MULTIDIMENSIONAL (SAE)')
casen22_ingresos=read_casen('INGRESOS')
casen22_escolaridad_15=read_casen('ESCOLARIDAD MAYORES 15')
casen22_escolaridad_18=read_casen('ESCOLARIDAD MAYORES 18')
casen22_escolaridad_18=read_casen('TASAS PARTICIPACIÓN LABORAL')
casen22_prevision=read_casen('PREVISIÓN DE SALUD')
casen22_migrantes=read_casen('MIGRANTES')
casen22_etnias=read_casen('ETNIAS')
censo17_rm = censo17[censo17['NOMBRE REGIÓN'] == 'METROPOLITANA DE SANTIAGO']
censo17_rm.rename(columns={'HOMBRES ':'HOMBRES'}, inplace=True)

#%%
# Streamlit: Seleccionar Comuna de interés
# Ordenar alfabéticamente las comunas y encontrar el índice de "SANTIAGO"
comunas = sorted(censo17_rm['NOMBRE COMUNA'].unique())
default_index = comunas.index('SANTIAGO') if 'SANTIAGO' in comunas else 0

# Streamlit: Seleccionar Comuna de interés
st.header('Región Metropolitana y sus comunas: Indicadores priorizados')
comuna = st.selectbox('Comunas:', options=comunas, index=default_index)

# Mostrar la comuna seleccionada
st.write('Comuna seleccionada:', comuna)

#%%
# import folium

# # Supongamos que tienes un DataFrame con latitudes y longitudes de las comunas
# # comuna_coords = pd.DataFrame({'Comuna': ['SANTIAGO', 'LAS CONDES', ...], 'Lat': [-33.45, -33.42, ...], 'Lon': [-70.65, -70.60, ...]})
# comuna_data = {
#     'Comuna': ['SANTIAGO', 'LAS CONDES', 'PROVIDENCIA', 'VITACURA', 'LA REINA', 'ÑUÑOA'],
#     'Lat': [-33.4372, -33.4248, -33.4363, -33.4009, -33.4421, -33.4543],
#     'Lon': [-70.6506, -70.5172, -70.6214, -70.6007, -70.5328, -70.6006]
# }

# # Crear el DataFrame
# comuna_coords = pd.DataFrame(comuna_data)

# # Obtén las coordenadas de la comuna seleccionada
# lat = comuna_coords.loc[comuna_coords['Comuna'] == comuna, 'Lat'].values[0]
# lon = comuna_coords.loc[comuna_coords['Comuna'] == comuna, 'Lon'].values[0]

# # Crea un mapa centrado en estas coordenadas
# mapa = folium.Map(location=[lat, lon], zoom_start=13)

# # Agrega un marcador para la comuna
# folium.Marker([lat, lon], popup=f'{selected_comuna}').add_to(mapa)

# Suponiendo que tienes un GeoJSON de las comunas de Santiago para las demarcaciones
# path_to_geojson = 'comunas_santiago.geojson'
# folium.GeoJson(path_to_geojson, name='geojson').add_to(mapa)

# Mostrar el mapa en Streamlit
# folium_static(mapa)

#%%

# Filtrar datos por comuna seleccionada para censo y CASEN
censo17_rm_comuna = censo17_rm[censo17_rm['NOMBRE COMUNA'] == comuna.upper()]
def casen_filtro_comuna(df):
    df['COMUNA'] = df['Comuna'].str.upper()
    df = df[df['COMUNA'] == comuna.upper()]
    df=df.reset_index()
    return df

casen22_pobrezam_com=casen_filtro_comuna(casen22_pobrezam)
casen22_ingresos_com=casen_filtro_comuna(casen22_ingresos)
casen22_prevision=casen_filtro_comuna(casen22_prevision)
#%%
# Preparación de datos para la pirámide poblacional
order = [
    "0 a 4", "5 a 9", "10 a 14", "15 a 19", "20 a 24", "25 a 29", "30 a 34",
    "35 a 39", "40 a 44", "45 a 49", "50 a 54", "55 a 59", "60 a 64", 
    "65 a 69", "70 a 74", "75 a 79", "80 a 84", "85 a 89", "90 a 94",
    "95 a 99", "100 o más"
]
censo17_rm_comuna_piramide = censo17_rm_comuna[censo17_rm_comuna['GRUPOS DE EDAD'] != "Total Comuna"]
censo17_rm_comuna_piramide['GRUPOS DE EDAD'] = pd.Categorical(censo17_rm_comuna_piramide['GRUPOS DE EDAD'], categories=order, ordered=True)
censo17_rm_comuna_piramide.sort_values('GRUPOS DE EDAD', inplace=True)

# Datos para la tabla de población
st.write('## Indicadores de territorio y demografía')
st.write('### Tabla de Población por Edad y Sexo')
st.dataframe(censo17_rm_comuna_piramide[['GRUPOS DE EDAD', 'HOMBRES', 'MUJERES']])

# Gráfico de la Pirámide Poblacional utilizando st.bar_chart
st.write('### Gráfico de la Pirámide Poblacional')
men_bins = []
women_bins = []

for age_group in order:
    group_data = censo17_rm_comuna[censo17_rm_comuna['GRUPOS DE EDAD'] == age_group]
    if not group_data.empty:
        men_count = group_data['HOMBRES'].iloc[0]
        women_count = group_data['MUJERES'].iloc[0]
        men_bins.append(men_count)
        women_bins.append(-women_count)  # Negativo para las mujeres para la pirámide
    else:
        men_bins.append(0)
        women_bins.append(0)

# Crear el gráfico
y = order
data = [
    go.Bar(y=y, x=men_bins, orientation='h', name='Men', hoverinfo='x'),
    go.Bar(y=y, x=women_bins, orientation='h', name='Women', text=-1 * np.array(women_bins).astype('int'), hoverinfo='text')
]
layout = go.Layout(
    yaxis=go.YAxis(title='Rango de edad'),
    xaxis=go.XAxis(
        title='Numero'
    ),
    barmode='overlay',
    bargap=0.1,
    title='Pirámide Poblacional',
    autosize=True  
)

fig = go.Figure(data=data, layout=layout)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)
#%%
# Distribución Urbana vs. Rural por Grupos de Edad
# st.write('## Distribución Urbana vs. Rural por Grupos de Edad')
# # Datos
# urban_population = []
# rural_population = []
# for group in order:
#     row = censo17_rm_comuna[censo17_rm_comuna['GRUPOS DE EDAD'] == group].iloc[0]
#     urban_population.append(row['TOTAL ÁREA URBANA'])
#     rural_population.append(row['TOTAL ÁREA RURAL'])

# # Crear el gráfico de Plotly
# fig = go.Figure()
# y = order
# data = [
#     go.Bar(y=y, x=urban_population, orientation='h', name='Urbano', hoverinfo='x'),
#     go.Bar(y=y, x=rural_population, orientation='h', name='Rural', hoverinfo='x')
# ]
# layout=go.Layout(
#     title='Distribución Urbana vs. Rural por Grupo de Edad',
#     barmode='group',
#     yaxis_title='Grupo de Edad',
#     xaxis_title='Población',
#     yaxis={'categoryorder': 'total ascending'},  # Opción para ordenar las categorías
#     template='plotly_white'
# )
# fig = go.Figure(data=data, layout=layout)
# st.plotly_chart(fig)

# st.write('## Distribución Urbana vs. Rural por Grupos de Edad')
# Datos
urban_population = []
rural_population = []
for group in order:
    row = censo17_rm_comuna[censo17_rm_comuna['GRUPOS DE EDAD'] == group].iloc[0]
    urban_population.append(row['TOTAL ÁREA URBANA'])
    rural_population.append(row['TOTAL ÁREA RURAL'])

fig = go.Figure(data=[
    go.Pie(labels=['Urbana', 'Rural'], values=[sum(urban_population), sum(rural_population)], pull=[0.05, 0])
])

fig.update_layout(title='Proporción de Población Urbana vs. Rural')
st.plotly_chart(fig)

#%%
st.write('## Indicadores Socioeconómicos: Multidimensional')
data_pobreza = {
    'Categoría': ['Pobres', 'No pobres'],
    'Porcentaje': [
        casen22_pobrezam_com.iloc[0]['Pobres'], 
        casen22_pobrezam_com.iloc[0]['No pobres']
    ]
}

# Crear el gráfico de torta
fig_pobreza = go.Figure(data=[
    go.Pie(
        labels=data_pobreza['Categoría'], 
        values=data_pobreza['Porcentaje'],
        pull=[0.05, 0],  # Resaltar la sección 'Pobres'
        # marker=dict(colors=['red', 'green'])  # Colores para las secciones
    )
])

fig_pobreza.update_layout(
    title='Distribución Multidimensional de Pobreza',
    # template='plotly_white'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig_pobreza)

#%%
# Datos para Indicadores Socioeconómicos: Ingresos con Plotly
st.write('## Indicadores Socioeconómicos: Ingresos')
data_ingresos = {
    'Categoría': ['Ingresos del trabajo', 'Ingreso Autónomo', 'Ingreso Monetario', 'Ingreso Total'],
    'Promedio': [
        casen22_ingresos_com.loc[0, 'Ingresos del trabajo'],
        casen22_ingresos_com.loc[0, 'Ingreso Autónomo'],
        casen22_ingresos_com.loc[0, 'Ingreso Monetario'],
        casen22_ingresos_com.loc[0, 'Ingreso Total']
    ]
}
fig_ingresos = go.Figure(data=[
    go.Bar(
        x=data_ingresos['Categoría'],
        y=data_ingresos['Promedio'],
        # marker=dict(color=['blue', 'blue', 'blue', 'blue'])  # Colores para cada categoría
    )
])
fig_ingresos.update_layout(
    title='Ingresos por Categoría',
    xaxis_title='Categoría',
    yaxis_title='Promedio',
    # template='plotly_white'
)
st.plotly_chart(fig_ingresos)

#%%
# Datos para el gráfico de torta para la comuna
labels = ['Fonasa', 'FF.AA. y del Orden', 'Isapre', 'Ninguno (Particular)', 'Otro Sistema', 'No Sabe']
values = [
    casen22_prevision.iloc[0]['fonasa'], 
    casen22_prevision.iloc[0]['ff.aa. y del orden'],
    casen22_prevision.iloc[0]['isapre'],
    casen22_prevision.iloc[0]['ninguno (particular)'],
    casen22_prevision.iloc[0]['otro sistema'],
    casen22_prevision.iloc[0]['no sabe']
]

# Crear el gráfico de torta
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=14)
fig.update_layout(
    title_text='Distribución de Tipos de Previsión',
    # template='plotly_white'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

