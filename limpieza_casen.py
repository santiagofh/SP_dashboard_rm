#%%
import pandas as pd
import numpy as np
from functools import reduce

path = 'data_raw/INDICADORES COMUNALES CASEN 2022 RMS.xlsx'

def read_casen(pestaña):
    df = pd.read_excel(path, sheet_name=pestaña, skiprows=4, nrows=52)
    df = df.replace('**', np.nan)
    df = df.replace('*', np.nan)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    # Exclude the 'Comuna' column from renaming to keep it as the merging key
    # df.columns = [col if col == 'Comuna' else col + suffix for col in df.columns]
    return df

# Load dataframes with respective suffixes
casen22_pobrezam = read_casen('POBREZA MULTIDIMENSIONAL (SAE)')
casen22_ingresos = read_casen('INGRESOS')
casen22_escolaridad_15 = read_casen('ESCOLARIDAD MAYORES 15')
casen22_escolaridad_18 = read_casen('ESCOLARIDAD MAYORES 18')
casen22_participacion_laboral = read_casen('TASAS PARTICIPACIÓN LABORAL')
casen22_prevision = read_casen('PREVISIÓN DE SALUD')
casen22_migrantes = read_casen('MIGRANTES')
casen22_etnias = read_casen('ETNIAS')

# data_frames = [
#     casen22_pobrezam, casen22_ingresos, casen22_escolaridad_15, 
#     casen22_escolaridad_18, casen22_participacion_laboral, casen22_prevision, 
#     casen22_migrantes, casen22_etnias
# ]

# # Merge all dataframes
# casen_combined = reduce(lambda left, right: pd.merge(left, right, on='Comuna', how='outer'), data_frames)

# print(casen_combined.head())

# casen_combined.to_csv('data_clean/CASEN_RM.csv')
casen22_pobrezam.to_csv('data_clean/casen22_pobrezam.csv')
casen22_ingresos.to_csv('data_clean/casen22_ingresos.csv')
casen22_escolaridad_15.to_csv('data_clean/casen22_escolaridad_15.csv')
casen22_escolaridad_18.to_csv('data_clean/casen22_escolaridad_18.csv')
casen22_participacion_laboral.to_csv('data_clean/casen22_participacion_laboral.csv')
casen22_prevision.to_csv('data_clean/casen22_prevision.csv')
casen22_migrantes.to_csv('data_clean/casen22_migrantes.csv')
casen22_etnias.to_csv('data_clean/casen22_etnias.csv')
# %%
