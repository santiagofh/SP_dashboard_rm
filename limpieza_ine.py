#%%
import pandas as pd
path_ine_proy='data_raw/ine_estimaciones-y-proyecciones-2002-2035_base-2017_comunas0381d25bc2224f51b9770a705a434b74.csv'
ine_proy=pd.read_csv(path_ine_proy, encoding='LATIN1')
#%%
ine_proy_rm=ine_proy.loc[ine_proy.Region==13]
# %%
ine_proy_rm.to_csv('data_clean/INE_Proyecciones_RM.csv')
# %%
