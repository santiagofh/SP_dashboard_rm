import pandas as pd
import numpy as np

# Define file paths
path17 = 'data_raw/INDICADORES COMUNALES CASEN 2017 RMS.xlsx'
path20 = 'data_raw/INDICADORES COMUNALES CASEN 2020 RMS.xlsx'
path22 = 'data_raw/INDICADORES COMUNALES CASEN 2022 RMS.xlsx'

# Function to read a sheet from a given Excel file
def read_casen(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=4, nrows=52)
    df = df.replace('**', np.nan)
    df = df.replace('*', np.nan)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    return df

# Function to read all sheets from a given Excel file and store in a dictionary
def read_all_sheets_to_dict(path, year, rename_dict):
    sheets = pd.ExcelFile(path).sheet_names
    data_dict = {}
    for sheet in sheets:
        df = read_casen(path, sheet)
        df['Año'] = year
        unified_sheet_name = rename_dict.get(sheet.strip(), sheet.strip())  # Use the mapped name if exists
        if unified_sheet_name in data_dict:
            data_dict[unified_sheet_name] = pd.concat([data_dict[unified_sheet_name], df], ignore_index=True)
        else:
            data_dict[unified_sheet_name] = df
    return data_dict

# Define a mapping of variant sheet names to unified names
rename_dict = {
    'POBREZA DE INGRESOS': 'POBREZA DE INGRESOS',
    'POBREZA DE INGRESOS (SAE)': 'POBREZA DE INGRESOS',
    ' POBREZA DE INGRESOS (SAE)': 'POBREZA DE INGRESOS',  # With space at the beginning
    'POBREZA MULTIDIMENSIONAL (SAE)': 'POBREZA MULTIDIMENSIONAL',
    'ÍNDICE DE HACINAMIENTO': 'HACINAMIENTO'
    # Add other mappings as needed
}

# Read data from all files into dictionaries
data_2017_dict = read_all_sheets_to_dict(path17, 2017, rename_dict)
data_2020_dict = read_all_sheets_to_dict(path20, 2020, rename_dict)
data_2022_dict = read_all_sheets_to_dict(path22, 2022, rename_dict)

# Combine all dictionaries into one and exclude 'ÍNDICE' and 'NOTAS'
combined_data_dict = {}
keys_to_exclude = ['ÍNDICE', 'NOTAS']

# List of all keys from all years
all_keys = set(data_2022_dict.keys()).union(data_2017_dict.keys()).union(data_2020_dict.keys())

for sheet in all_keys:
    if sheet not in keys_to_exclude:
        combined_df = pd.concat(
            [
                data_2017_dict.get(sheet, pd.DataFrame()), 
                data_2020_dict.get(sheet, pd.DataFrame()), 
                data_2022_dict.get(sheet, pd.DataFrame())],
            ignore_index=True
        )
        combined_df['sheet'] = sheet
        combined_data_dict[sheet] = combined_df

# Convert each DataFrame in combined_data_dict to a JSON string
json_data_dict = {sheet: df.to_dict(orient='records') for sheet, df in combined_data_dict.items()}

# Save the combined JSON data to a single file
with open('data_clean/casen_17_22.json', 'w') as json_file:
    import json
    json.dump(json_data_dict, json_file, indent=4)

print("Archivo JSON único creado con éxito.")
