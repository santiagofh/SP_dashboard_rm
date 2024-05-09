#%%
import pandas as pd
path='data_raw/1_1_poblacion.xls'
df=pd.read_excel(path, sheet_name='Comuna',skiprows=2)
#%%
df_rm=df.loc[df['CÓDIGO REGIÓN']=='13']
# %%
df_rm.to_csv('data_clean/CENSO17_Poblacion_rm.csv')
# %%
