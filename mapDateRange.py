import numpy as np
import sys
import pandas as pd
import xarray as xr
import os
import torch
import netCDF4
import json
import matplotlib.pyplot as plt
import datetime as dt
import time
from scipy import ndimage
from configparser import ConfigParser
from findMatrixCoordsBedrock import *
from nasnet_mobile import *
from checkMODISBounds import *
import scipy.spatial.qhull as qhull

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def interp_weights(xyz, uvw):
	d = uvw.shape[1]
	tri = qhull.Delaunay(xyz)
	simplex = tri.find_simplex(uvw)
	vertices = np.take(tri.simplices, simplex, axis=0)
	temp = np.take(tri.transform, simplex, axis=0)
	delta = uvw - temp[:, d]
	bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
	return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, fill_value=np.nan):
	ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
	ret[np.any(wts < 0, axis=1)] = fill_value
	return ret

startDate = pd.Timestamp(year=2018, month=7, day=5, hour=0)
endDate = pd.Timestamp(year=2018, month=10, day=31, hour=0)

dates = []

testDate = startDate
while(testDate <= endDate):
	dates.append(testDate)
	testDate = testDate + pd.Timedelta(days=1)

np.save('map_images/dates.npy', dates)

computeData = False
if(computeData == True):

	step_size = 0.0089
	florida_x = np.arange(-92, -75, step_size)
	florida_y = np.arange(20, 35, step_size)
	xx, yy = np.meshgrid(florida_x, florida_y)
	xi = np.zeros((florida_x.shape[0]*florida_y.shape[0], 2))
	xi[:, 0] = xx.flatten()
	xi[:, 1] = yy.flatten()

	# 'par', count, 'Kd_490', count, 'chlor_a', count, 'Rrs_443', count, 'Rrs_469', count, 'Rrs_488', count, 'nflh', count
	florida_stats = np.zeros((florida_x.shape[0], florida_y.shape[0], 14, len(dates)))
	florida_lats = np.tile(florida_y, len(florida_x))
	florida_lons = np.repeat(florida_x, len(florida_y))

	minLonIdx = find_nearest(florida_x, -85)
	maxLonIdx = find_nearest(florida_x, -80)

	minLatIdx = find_nearest(florida_y, 24)
	maxLatIdx = find_nearest(florida_y, 30)

	data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
	data_list = os.listdir(data_folder)

	reduced_file_list = []
	for i in range(len(data_list)):
		if(i%100 == 0):
			print('Getting file dates: {}/{}'.format(i, len(data_list)))
		file_id = data_list[i]
		file_path = data_folder + '/' + file_id

		fh = netCDF4.Dataset(file_path, mode='r')
		collectionDate = fh.time_coverage_start[0:10]
		collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

		if(collectionTimeStamp >= (startDate - pd.Timedelta(days=1)) and collectionTimeStamp <= (endDate + pd.Timedelta(days=1))):
			reduced_file_list.append(data_list[i])

	days_of_imagery_to_average = 14

	for i in range(len(reduced_file_list)):
		print('Processing files: {}/{}'.format(i, len(reduced_file_list)))

		file_id = reduced_file_list[i]
		file_path = data_folder + '/' + file_id

		fh = netCDF4.Dataset(file_path, mode='r')
		collectionDate = fh.time_coverage_start[0:10]
		collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

		days_ahead = [(date_temp - collectionTimeStamp).days for date_temp in dates]

		if(any(day_ahead >= 0 and day_ahead <= days_of_imagery_to_average for day_ahead in days_ahead)):
			dataset = xr.open_dataset(file_path, 'geophysical_data')
			nav_dataset = xr.open_dataset(file_path, 'navigation_data')
			latitude = nav_dataset['latitude']
			longitude = nav_dataset['longitude']
			latarr = np.array(latitude).flatten()
			longarr = np.array(longitude).flatten()

			par = np.array(dataset['par']).flatten()
			Kd_490 = np.array(dataset['Kd_490']).flatten()
			chlor_a = np.array(dataset['chlor_a']).flatten()
			Rrs_443 = np.array(dataset['Rrs_443']).flatten()
			Rrs_469 = np.array(dataset['Rrs_469']).flatten()
			Rrs_488 = np.array(dataset['Rrs_488']).flatten()
			nflh = np.array(dataset['nflh']).flatten()

			par = checkMODISBounds(par, 'par')
			Kd_490 = checkMODISBounds(Kd_490, 'Kd_490')
			chlor_a = checkMODISBounds(chlor_a, 'chlor_a')
			Rrs_443 = checkMODISBounds(Rrs_443, 'Rrs_443')
			Rrs_469 = checkMODISBounds(Rrs_469, 'Rrs_469')
			Rrs_488 = checkMODISBounds(Rrs_488, 'Rrs_488')
			nflh = checkMODISBounds(nflh, 'nflh')

			datapoints = np.zeros((len(longarr), 2))
			datapoints[:, 0] = longarr
			datapoints[:, 1] = latarr

			day_inds = [day_ahead >= 0 and day_ahead <= days_of_imagery_to_average for day_ahead in days_ahead]
			day_inds = np.where(np.array(day_inds) == True)[0]

			vtx, wts = interp_weights(datapoints, xi)
			interpolated_data = interpolate(par, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 0, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 1, day_inds[j]] += ~np.isnan(interpolated_data)

			interpolated_data = interpolate(Kd_490, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 2, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 3, day_inds[j]] += ~np.isnan(interpolated_data)

			interpolated_data = interpolate(chlor_a, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 4, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 5, day_inds[j]] += ~np.isnan(interpolated_data)

			interpolated_data = interpolate(Rrs_443, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 6, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 7, day_inds[j]] += ~np.isnan(interpolated_data)

			interpolated_data = interpolate(Rrs_469, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 8, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 9, day_inds[j]] += ~np.isnan(interpolated_data)

			interpolated_data = interpolate(Rrs_488, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 10, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 11, day_inds[j]] += ~np.isnan(interpolated_data)

			interpolated_data = interpolate(nflh, vtx, wts)
			interpolated_data = np.reshape(interpolated_data, (florida_x.shape[0], florida_y.shape[0]), order='F')
			for j in range(len(day_inds)):
				florida_stats[:, :, 12, day_inds[j]] += np.nan_to_num(interpolated_data)
				florida_stats[:, :, 13, day_inds[j]] += ~np.isnan(interpolated_data)

	features = np.zeros((florida_stats.shape[0], florida_stats.shape[1], 8, len(dates)))
	for i in range(florida_stats.shape[0]):
		for j in range(florida_stats.shape[1]):
			for k in range(len(dates)):
				if(florida_stats[i, j, 1, k] == 0):
					features[i, j, 0, k] = 0
				else:
					features[i, j, 0, k] = florida_stats[i, j, 0, k]/florida_stats[i, j, 1, k]

				if(florida_stats[i, j, 3, k] == 0):
					features[i, j, 1, k] = 0
				else:
					features[i, j, 1, k] = florida_stats[i, j, 2, k]/florida_stats[i, j, 3, k]

				if(florida_stats[i, j, 5, k] == 0):
					features[i, j, 2, k] = 0
				else:
					features[i, j, 2, k] = florida_stats[i, j, 4, k]/florida_stats[i, j, 5, k]

				if(florida_stats[i, j, 7, k] == 0):
					features[i, j, 3, k] = 0
				else:
					features[i, j, 3, k] = florida_stats[i, j, 6, k]/florida_stats[i, j, 7, k]

				if(florida_stats[i, j, 9, k] == 0):
					features[i, j, 4, k] = 0
				else:
					features[i, j, 4, k] = florida_stats[i, j, 8, k]/florida_stats[i, j, 9, k]

				if(florida_stats[i, j, 11, k] == 0):
					features[i, j, 5, k] = 0
				else:
					features[i, j, 5, k] = florida_stats[i, j, 10, k]/florida_stats[i, j, 11, k]

				if(florida_stats[i, j, 13, k] == 0):
					features[i, j, 6, k] = 0
				else:
					features[i, j, 6, k] = florida_stats[i, j, 12, k]/florida_stats[i, j, 13, k]

	np.save('/run/media/rfick/UF10/features.npy', features)
else:
	features = np.load('/run/media/rfick/UF10/features.npy')

	step_size = 0.0089
	florida_x = np.arange(-92, -75, step_size)
	florida_y = np.arange(20, 35, step_size)
	florida_lats = np.tile(florida_y, len(florida_x))
	florida_lons = np.repeat(florida_x, len(florida_y))

	minLonIdx = find_nearest(florida_x, -85)
	maxLonIdx = find_nearest(florida_x, -80)

	minLatIdx = find_nearest(florida_y, 24)
	maxLatIdx = find_nearest(florida_y, 30)

	days_of_imagery_to_average = 14

bedrock_x = np.load('florida_x.npy')
bedrock_y = np.load('florida_y.npy')
bedrock_z = np.load('florida_z.npy')

original_size = features.shape

features = np.reshape(features, (features.shape[0]*features.shape[1], 8, len(dates)))

bedrock_indices = findMatrixCoordsBedrock(bedrock_x, bedrock_y, florida_lons, florida_lats)
bedrock_indices = np.array(bedrock_indices)

for j in range(len(dates)):
	features[:, 7, j] = bedrock_z[bedrock_indices[:, 0], bedrock_indices[:, 1]]

features = np.reshape(features, original_size)

features = np.moveaxis(features, 1, 0)

land_mask = features[minLatIdx:maxLatIdx,minLonIdx:maxLonIdx,7,0] >= 0

features = np.moveaxis(features, 2, 0)

for day_counter in range(days_of_imagery_to_average, len(dates)):
	day = dates[day_counter]
	print('Processing days: {}/{}'.format(day_counter, len(dates)))

	day_features = np.squeeze(features[:, :, :, day_counter])

	#data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/red_tide_conv_selected_data_correct_abovebelow50000/'
	#np.save(data_folder+'day_features.npy', day_features)

	model = NASNetAMobile(num_classes=2).cuda()
	model.load_state_dict(torch.load('saved_model_info/model_correct.pt'))
	model.eval()

	red_tide = np.zeros((maxLatIdx-minLatIdx, maxLonIdx-minLonIdx))

	for i in range(minLatIdx, maxLatIdx):
		print('Processing row {}/{}'.format(i-minLatIdx, maxLatIdx-minLatIdx))
		for j in range(minLonIdx, maxLonIdx):
			image_features = np.zeros((1, features.shape[0], 224, 224))
			image_features[0, :, 62:163, 62:163] = day_features[:, i-50:i+51, j-50:j+51]
			featureTensor = torch.tensor(image_features).float().cuda()

			outputFeatures, output = model(featureTensor)

			red_tide[i-minLatIdx, j-minLonIdx] = output[0, 1].detach().cpu().numpy()

	red_tide[land_mask] = -1

	np.save('map_images/red_tide{}.npy'.format(day_counter), red_tide)

	plt.figure(dpi=500)
	plt.imshow(red_tide)
	plt.clim(-1, 1)
	plt.colorbar()
	plt.title('Red Tide Prediction {}/{}/{}'.format(day.month, day.day, day.year))
	plt.gca().invert_yaxis()
	plt.savefig('map_images/red_tide_image{}.png'.format(str(day_counter - days_of_imagery_to_average).zfill(5)), bbox_inches='tight')