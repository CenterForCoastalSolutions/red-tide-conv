import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

positive_locations = np.load('convDataPlotting/positive_locations.npy')
positive_data_lon = positive_locations[:, 0]
positive_data_lat = positive_locations[:, 1]
positive_data_concs = np.load('convDataPlotting/positive_concs.npy')
negative_locations = np.load('convDataPlotting/negative_locations.npy')
negative_data_lon = negative_locations[:, 0]
negative_data_lat = negative_locations[:, 1]
negative_data_concs = np.load('convDataPlotting/negative_concs.npy')

data_lon = np.concatenate((negative_data_lon, positive_data_lon))
data_lat = np.concatenate((negative_data_lat, positive_data_lat))
data_concs = np.concatenate((negative_data_concs, positive_data_concs))

df = pd.read_excel(file_path, engine='openpyxl')
df_dates = df['Sample Date'].tolist()
df_depths = df['Sample Depth (m)'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()
df_concs = df['Karenia brevis abundance (cells/L)']

background_lon = data_lon[(data_concs >= 0) & (data_concs < 1000)]
background_lat = data_lat[(data_concs >= 0) & (data_concs < 1000)]

verylow_lon = data_lon[(data_concs >= 1000) & (data_concs < 10000)]
verylow_lat = data_lat[(data_concs >= 1000) & (data_concs < 10000)]

low_lon = data_lon[(data_concs >= 10000) & (data_concs < 100000)]
low_lat = data_lat[(data_concs >= 10000) & (data_concs < 100000)]

medium_lon = data_lon[(data_concs >= 100000) & (data_concs < 1000000)]
medium_lat = data_lat[(data_concs >= 100000) & (data_concs < 1000000)]

high_lon = data_lon[(data_concs >= 1000000)]
high_lat = data_lat[(data_concs >= 1000000)]

plt.figure(figsize=(8, 8))
m = Basemap(projection='gnom', resolution='h', lat_0=26.3, lon_0=-82.5, width=400000, height=500000)
m.fillcontinents(color="#FFDDCC", lake_color="#DDEEFF")
m.drawmapboundary(fill_color="#DDEEFF")
m.drawcoastlines()
m.scatter(background_lon, background_lat, latlon=True, s=10, c='grey', marker='.')
m.scatter(verylow_lon, verylow_lat, latlon=True, s=10, c='white', marker='.')
m.scatter(low_lon, low_lat, latlon=True, s=10, c='yellow', marker='.')
m.scatter(medium_lon, medium_lat, latlon=True, s=10, c='orange', marker='.')
m.scatter(high_lon, high_lat, latlon=True, s=10, c='red', marker='.')
plt.title('In situ Sample Locations')

plt.savefig('inSituMap.png', bbox_inches='tight')