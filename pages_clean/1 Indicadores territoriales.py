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
path_geo='data/Chile_comunas_20230405.geojson'
path_ine_proy='data/estimaciones-y-proyecciones-2002-2035-comuna-y-área-urbana-y-rural11df0b16cde04242827bef3fd62529c5.xlsx'

#%%
# READ DATA
ine17=pd.read_excel(path_ine_proy)
censo17 = pd.read_excel(path_censo17, sheet_name='Comuna', skiprows=2)
gdf = gpd.read_file(path_geo)
#%%
censo17_rm = censo17[censo17['NOMBRE REGIÓN'] == 'METROPOLITANA DE SANTIAGO']
censo17_rm.rename(columns={'HOMBRES ':'HOMBRES'}, inplace=True)
ine17_rm=ine17[ine17['Nombre Region'] == 'Metropolitana de Santiago']
#%%

#%%
comunas = sorted(censo17_rm['NOMBRE COMUNA'].unique())
default_index = comunas.index('SANTIAGO') if 'SANTIAGO' in comunas else 0
st.header('Región Metropolitana y sus comunas: Indicadores priorizados')
comuna = st.selectbox('Comunas:', options=comunas, index=default_index)
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
ine17_rm['Nombre Comuna'] = ine17_rm['Nombre Comuna'].str.upper()
ine17_rm_comuna=ine17_rm[ine17_rm['Nombre Comuna'] == comuna.upper()]
gdf['Comuna'] = gdf['Comuna'].str.upper() 
gdf_comuna = gdf[gdf['Comuna'] == comuna.upper()]

#%%
st.write(("## Visualizar mapa de la comuna"))

if not gdf_comuna.empty:
    centroid = gdf_comuna.geometry.centroid.iloc[0]
    map_data = pd.DataFrame({
        'lat': [centroid.y],
        'lon': [centroid.x],
        'color':None
    })
    st.map(map_data)
else:
    st.write("No se encontró la comuna seleccionada en los datos geográficos.")
if st.button("Visualizar mapa con los limites de la comuna"):
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
#%%
st.write('## Superficie y densidad poblacional')


#%%
st.write('## Distribución urbana vs. rural')
# Datos
order = [
    "0 a 4", "5 a 9", "10 a 14", "15 a 19", "20 a 24", "25 a 29", "30 a 34",
    "35 a 39", "40 a 44", "45 a 49", "50 a 54", "55 a 59", "60 a 64", 
    "65 a 69", "70 a 74", "75 a 79", "80 a 84", "85 a 89", "90 a 94",
    "95 a 99", "100 o más"
]
urban_population = []
rural_population = []
for group in order:
    row = censo17_rm_comuna[censo17_rm_comuna['GRUPOS DE EDAD'] == group].iloc[0]
    urban_population.append(row['TOTAL ÁREA URBANA'])
    rural_population.append(row['TOTAL ÁREA RURAL'])

fig = go.Figure(data=[
    go.Pie(labels=['Urbana', 'Rural'], values=[sum(urban_population), sum(rural_population)], pull=[0.05, 0])
])

fig.update_layout(title='Proporción de población urbana vs. rural')
st.plotly_chart(fig)
st.write('_Fuente: Elaboración propia a partir de CENSO 2017_')
#%%
