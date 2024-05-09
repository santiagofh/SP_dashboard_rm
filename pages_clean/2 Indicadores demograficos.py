#%%
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
# Rutas de datos
path_censo17 = 'data/1_2_POBLACION.xlsx'
path_casen22 = 'data/INDICADORES COMUNALES CASEN 2022 RMS.xlsx'
# path_geo='data/Chile_comunas_20230405.geojson'
path_ine_proy='data/estimaciones-y-proyecciones-2002-2035-comuna-y-área-urbana-y-rural11df0b16cde04242827bef3fd62529c5.xlsx'
#%%
#Funciones READ
def read_casen(pestaña):
    df=pd.read_excel(path_casen22, sheet_name=pestaña, skiprows=4)
    df=df.replace('**',np.nan)
    df=df.replace('*',np.nan)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True) 
    return df

#%%
# READ DATA
ine17=pd.read_excel(path_ine_proy)
censo17 = pd.read_excel(path_censo17, sheet_name='Comuna', skiprows=2)
casen22_pobrezam=read_casen('POBREZA MULTIDIMENSIONAL (SAE)')
casen22_ingresos=read_casen('INGRESOS')
casen22_escolaridad_15=read_casen('ESCOLARIDAD MAYORES 15')
casen22_escolaridad_18=read_casen('ESCOLARIDAD MAYORES 18')
casen22_escolaridad_18=read_casen('TASAS PARTICIPACIÓN LABORAL')
casen22_prevision=read_casen('PREVISIÓN DE SALUD')
casen22_migrantes=read_casen('MIGRANTES')
casen22_etnias=read_casen('ETNIAS')
# gdf = gpd.read_file(path_geo)
#%%
censo17_rm = censo17[censo17['NOMBRE REGIÓN'] == 'METROPOLITANA DE SANTIAGO']
censo17_rm.rename(columns={'HOMBRES ':'HOMBRES'}, inplace=True)
ine17_rm=ine17[ine17['Nombre Region'] == 'Metropolitana de Santiago']
#%%

#%%
# Streamlit: Seleccionar Comuna de interés
# Ordenar alfabéticamente las comunas y encontrar el índice de "SANTIAGO"
comunas = sorted(censo17_rm['NOMBRE COMUNA'].unique())
default_index = comunas.index('SANTIAGO') if 'SANTIAGO' in comunas else 0

# Streamlit: Seleccionar Comuna de interés
st.header('Región Metropolitana y sus comunas: Indicadores priorizados')
comuna = st.selectbox('Comunas:', options=comunas, index=default_index)

# Mostrar la comuna seleccionada
st.write('## Comuna seleccionada:', comuna)

#%%

#%%
# Filtrar datos por comuna seleccionada 
def casen_filtro_comuna(df):
    df['COMUNA'] = df['Comuna'].str.upper()
    df = df[df['COMUNA'] == comuna.upper()]
    df=df.reset_index()
    return df

censo17_rm_comuna = censo17_rm[censo17_rm['NOMBRE COMUNA'] == comuna.upper()]
casen22_pobrezam_comuna=casen_filtro_comuna(casen22_pobrezam)
casen22_ingresos_comuna=casen_filtro_comuna(casen22_ingresos)
casen22_prevision_comuna=casen_filtro_comuna(casen22_prevision)
ine17_rm['Nombre Comuna'] = ine17_rm['Nombre Comuna'].str.upper()
ine17_rm_comuna=ine17_rm[ine17_rm['Nombre Comuna'] == comuna.upper()]

#%%
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

# Creación de dos columnas

st.subheader('Indicadores de territorio y demografía')
# Tabla de población por edad y sexo
st.dataframe(censo17_rm_comuna_piramide[['GRUPOS DE EDAD', 'HOMBRES', 'MUJERES']])
st.caption('Fuente: Elaboración propia a partir de CENSO 2017')


st.subheader('Gráfico de la pirámide poblacional')
# Gráfico de la Pirámide Poblacional utilizando st.plotly_chart
men_bins = []
women_bins = []

for age_group in order:
    group_data = censo17_rm_comuna[censo17_rm_comuna['GRUPOS DE EDAD'] == age_group]
    if not group_data.empty:
        men_count = group_data['HOMBRES'].iloc[0]
        women_count = group_data['MUJERES'].iloc[0]
        men_bins.append(men_count)
        women_bins.append(-women_count)  # Negativo para las mujeres para la pirámide

y = order
data = [
    go.Bar(y=y, x=men_bins, orientation='h', name='Men', hoverinfo='x'),
    go.Bar(y=y, x=women_bins, orientation='h', name='Women', text=-1 * np.array(women_bins).astype('int'), hoverinfo='text')
]
layout = go.Layout(
    yaxis=go.YAxis(title='Rango de edad'),
    xaxis=go.XAxis(
        title='Número'
    ),
    barmode='overlay',
    bargap=0.1,
    title='Pirámide Poblacional',
    autosize=True
)

fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig)
st.caption('Fuente: Elaboración propia a partir de CENSO 2017')

#%%
# Población proyectada
st.write('## Poblacion proyectada')
population_data = ine17_rm_comuna[['Nombre Comuna', 'Sexo (1=Hombre 2=Mujer)'] + [f'Poblacion {year}' for year in range(2002, 2036)]]
population_data_melted = population_data.melt(id_vars=['Nombre Comuna', 'Sexo (1=Hombre 2=Mujer)'], var_name='Year', value_name='Population')
population_data_melted['Year'] = population_data_melted['Year'].str.extract('(\d+)').astype(int)
total_population_by_gender = population_data_melted.groupby(['Year', 'Sexo (1=Hombre 2=Mujer)'])['Population'].sum().reset_index()
total_population_by_gender.sort_values(by=['Year', 'Sexo (1=Hombre 2=Mujer)'], inplace=True)
total_population_by_gender['Sexo (1=Hombre 2=Mujer)'] = total_population_by_gender['Sexo (1=Hombre 2=Mujer)'].map({1: 'Male', 2: 'Female'})
fig = px.line(
    total_population_by_gender,
    x='Year',
    y='Population',
    color='Sexo (1=Hombre 2=Mujer)',
    title='Total Projected Population by Gender for ' + comuna,
    labels={'Sexo (1=Hombre 2=Mujer)': 'Gender', 'Year': 'Year', 'Population': 'Population'}
)
current_year = datetime.now().year
fig.add_vline(x=current_year, line_width=2, line_dash="dash", line_color="red")
fig.update_layout(
    yaxis_title='Total Population',
    xaxis_title='Year',
    legend_title='Gender',

)
st.plotly_chart(fig)
st.write('_Fuente: Elaboración propia a partir de CENSO 2017_')
#%% 
# Porcentaje de la comuna sobre el total de la región

# Data Filtering
total_region = censo17_rm[censo17_rm['GRUPOS DE EDAD'] == 'Total Comuna']
total_population_region = total_region['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA'].sum()
total_comuna_population = total_region[total_region['NOMBRE COMUNA'] == comuna]['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA'].values[0]

# Calculations
percentage_comuna = (total_comuna_population / total_population_region) * 100

# Dataframe Transformation
comuna_populations = total_region[['NOMBRE COMUNA', 'TOTAL POBLACIÓN EFECTIVAMENTE CENSADA']]
comuna_populations['Percentage'] = (comuna_populations['TOTAL POBLACIÓN EFECTIVAMENTE CENSADA'] / total_population_region) * 100

# Sorting Data
comuna_populations = comuna_populations.sort_values('Percentage', ascending=False).reset_index(drop=True)

# Visualization
st.write('## Porcentaje de la comuna sobre el total de la región')
st.write(f"Total población de la región Metropolitana de Santiago: `{total_population_region:,}`")
st.write(f"Total población de {comuna}: `{total_comuna_population:,}`")
st.write(f"Porcentaje de {comuna} sobre el total de la región: `{percentage_comuna:.2f}%`")
progress = percentage_comuna / 100
st.progress(progress)


# Update colors list with a specific RGB value
highlight_color = 'rgb(255, 0, 0)'  # Red color for the selected comuna
default_color = 'rgb(0, 104, 201)'  # Blue color for all other comunas
colors = [highlight_color if row['NOMBRE COMUNA'] == comuna else default_color for index, row in comuna_populations.iterrows()]

fig = px.pie(comuna_populations,
            values='Percentage',
            names='NOMBRE COMUNA',
            title='Population Distribution in the Santiago Metropolitan Region',
            color_discrete_sequence=[highlight_color, default_color])
fig.update_traces(marker=dict(colors=colors), textinfo='none')
st.plotly_chart(fig)



#%%
st.write('## Indicadores Socioeconómicos: Multidimensional')
data_pobreza = {
    'Categoría': ['Pobres', 'No pobres'],
    'Porcentaje': [
        casen22_pobrezam_comuna.iloc[0]['Pobres'], 
        casen22_pobrezam_comuna.iloc[0]['No pobres']
    ]
}

fig_pobreza = go.Figure(data=[
    go.Pie(
        labels=data_pobreza['Categoría'], 
        values=data_pobreza['Porcentaje'],
        pull=[0.05, 0],
    )
])

fig_pobreza.update_layout(
    title='Distribución Multidimensional de Pobreza',
)

st.plotly_chart(fig_pobreza)
st.write('_Fuente: Elaboración propia a partir de CASEN 2017_')
#%%
st.write('# Datos para Indicadores Socioeconómicos')
#%%
st.write('## Indicadores Socioeconómicos: Ingresos')
data_ingresos = {
    'Categoría': ['Ingresos del trabajo', 'Ingreso Autónomo', 'Ingreso Monetario', 'Ingreso Total'],
    'Promedio': [
        casen22_ingresos_comuna.loc[0, 'Ingresos del trabajo'],
        casen22_ingresos_comuna.loc[0, 'Ingreso Autónomo'],
        casen22_ingresos_comuna.loc[0, 'Ingreso Monetario'],
        casen22_ingresos_comuna.loc[0, 'Ingreso Total']
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
st.write('_Fuente: Elaboración propia a partir de CASEN 2017_')

#%%
# Datos para el gráfico de torta para la comuna
labels = ['Fonasa', 'FF.AA. y del Orden', 'Isapre', 'Ninguno (Particular)', 'Otro Sistema', 'No Sabe']
values = [
    casen22_prevision_comuna.iloc[0]['fonasa'], 
    casen22_prevision_comuna.iloc[0]['ff.aa. y del orden'],
    casen22_prevision_comuna.iloc[0]['isapre'],
    casen22_prevision_comuna.iloc[0]['ninguno (particular)'],
    casen22_prevision_comuna.iloc[0]['otro sistema'],
    casen22_prevision_comuna.iloc[0]['no sabe']
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

