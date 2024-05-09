import geopandas as gpd

# Suponiendo que 'comunas_rm' es tu GeoDataFrame que deseas exportar
comunas_rm = gpd.read_file('data_raw/Chile_comunas_20230405.geojson')
comunas_rm = comunas_rm.loc[comunas_rm.codregion == 13]

# Exportar a GeoJSON
comunas_rm.to_file('data_clean/Poligonos_comunas_RM.geojson', driver='GeoJSON')
