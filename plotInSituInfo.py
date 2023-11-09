import pandas as pd
import numpy as np

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')
df_dates = df['Sample Date'].tolist()
df_depths = df['Sample Depth (m)'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()
df_counties = df['County'].tolist()
df_concs = df['Karenia brevis abundance (cells/L)']

all_years = []

for date in df_dates:
	all_years.append(date.year)

print(min(df_dates))
print(max(df_dates))

print(len(df_dates))

county_set = set(df_counties)
county_set = list(county_set)
print(county_set)

county_counts = np.zeros((len(county_set), 1))
for county in df_counties:
	ind = county_set.index(county)
	county_counts[ind] += 1

print(county_counts)
