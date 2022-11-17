import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import scandir

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')
df_dates = df['Sample Date'].tolist()
df_depths = df['Sample Depth (m)'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()
df_concs = df['Karenia brevis abundance (cells/L)']

inputTopLevelFolder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/red_tide_conv_images_correct'
save_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/'

subfolders = [ f.path for f in scandir(inputTopLevelFolder) if f.is_dir() ]

data = np.zeros((23957, 101, 101, 23))
concs = np.zeros((23957))
dates = []

valid_count = 0

for i in range(len(subfolders)):
	if(i%100 == 0):
		print('{}/{}'.format(i, len(subfolders)))
	subfolder = subfolders[i]

	imagery = np.load(subfolder+'/imagery.npy')
	conc = np.load(subfolder+'/conc.npy')
	date = np.load(subfolder+'/date.npy', allow_pickle=True)

	imagery = np.reshape(imagery, (101, 101, 23))

	if(np.isnan(imagery[50, 50, 2]) == False):
		data[valid_count, :, :, :] = imagery
		concs[valid_count] = conc
		dates.append(date)
		
		valid_count = valid_count + 1

np.save(save_folder+'red_tide_conv_images_correct.npy', data)
np.save(save_folder+'red_tide_conv_concs_correct.npy', concs)
np.save(save_folder+'red_tide_conv_dates_correct.npy', dates)