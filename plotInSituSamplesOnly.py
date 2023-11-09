import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

startDate = pd.Timestamp(year=2006, month=7, day=1, hour=0)
endDate = pd.Timestamp(year=2006, month=12, day=31, hour=0)

dates = []

testDate = startDate
while(testDate <= endDate):
	dates.append(testDate)
	testDate = testDate + pd.Timedelta(days=1)

step_size = 0.0089
florida_x = np.arange(-92, -75, step_size)
florida_y = np.arange(20, 35, step_size)

minLonIdx = find_nearest(florida_x, -85)
maxLonIdx = find_nearest(florida_x, -80)

minLatIdx = find_nearest(florida_y, 24)
maxLatIdx = find_nearest(florida_y, 30)

florida_x = florida_x[minLonIdx:maxLonIdx]
florida_y = florida_y[minLatIdx:maxLatIdx]

florida_x = florida_x[220:440]
florida_y = florida_y[50:470]

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')

df_dates = df['Sample Date'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()
df_concs = df['Karenia brevis abundance (cells/L)']

red_tide = np.load('case_study/red_tide_output21.npy')
red_tide[red_tide > -0.9] = 0

daycounter = 0

for day in dates:

	collectionTimeStamp = day
	sample_data_inds = [j for j in range(len(df_dates)) if (collectionTimeStamp - df_dates[j]).days <= 5 and (collectionTimeStamp - df_dates[j]).days >= 0]

	plt.figure(dpi=500)
	ax = plt.gca()
	plt.imshow(red_tide[220:440, 50:470].T)
	plt.clim(-1, 1)
	#plt.colorbar()
	plt.title(day.strftime('%Y-%m-%d'))
	plt.gca().invert_yaxis()
	plt.axis('off')

	for i in range(len(sample_data_inds)):
		opacity = 125/255
		if(df_concs[sample_data_inds[i]] < 1000):
			colorToUse = (128/255, 128/255, 128/255, opacity)
			edgeColorToUse = (0, 0, 0, opacity)
			radius = 3
		elif(df_concs[sample_data_inds[i]] > 1000 and df_concs[sample_data_inds[i]] < 10000):
			colorToUse = (255/255, 255/255, 255/255, opacity)
			edgeColorToUse = (255/255, 255/255, 255/255, opacity)
			radius = 6
		elif(df_concs[sample_data_inds[i]] > 10000 and df_concs[sample_data_inds[i]] < 100000):
			colorToUse = (255/255, 255/255, 0/255, opacity)
			edgeColorToUse = (255/255, 255/255, 0/255, opacity)
			radius = 9
		elif(df_concs[sample_data_inds[i]] > 100000 and df_concs[sample_data_inds[i]] < 1000000):
			colorToUse = (255/255, 165/255, 0/255, opacity)
			edgeColorToUse = (255/255, 165/255, 0/255, opacity)
			radius = 12
		elif(df_concs[sample_data_inds[i]] > 1000000):
			colorToUse = (255/255, 0/255, 0/255, opacity)
			edgeColorToUse = (255/255, 0/255, 0/255, opacity)
			radius = 15
		x_coord = find_nearest(florida_x, df_lons[sample_data_inds[i]])
		y_coord = find_nearest(florida_y, df_lats[sample_data_inds[i]])
		circle = plt.Circle((x_coord, y_coord), 3, facecolor=colorToUse)
		ax.add_patch(circle)

	plt.savefig('in-situ-samples-only/'+"{:05d}".format(daycounter)+'.png', bbox_inches='tight')

	daycounter += 1