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
from convertFeaturesByDepth import *
from findMatrixCoordsBedrock import *
from pixelwise_model import *
from utils import *
import scipy.spatial.qhull as qhull

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

startDate = pd.Timestamp(year=2006, month=6, day=1, hour=0)
endDate = pd.Timestamp(year=2006, month=9, day=30, hour=0)
#startDate = pd.Timestamp(year=2006, month=9, day=15, hour=0)
#endDate = pd.Timestamp(year=2006, month=12, day=31, hour=0)

dates = []

save_folder = 'map_images/'

testDate = startDate
while(testDate <= endDate):
	dates.append(testDate)
	testDate = testDate + pd.Timedelta(days=1)

np.save(save_folder+'/dates.npy', dates)

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

data_folder = '/data-drive-12521/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

days_of_imagery_to_average = 10

reduced_file_list = []
for i in range(len(data_list)):
	if(i%100 == 0):
		print('Getting file dates: {}/{}'.format(i, len(data_list)))
	file_id = data_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	if(collectionTimeStamp >= (startDate - pd.Timedelta(days=days_of_imagery_to_average)) and collectionTimeStamp <= (endDate + pd.Timedelta(days=1))):
		reduced_file_list.append(data_list[i])

for i in range(len(reduced_file_list)):
	print('Processing files: {}/{}'.format(i, len(reduced_file_list)))

	file_id = reduced_file_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)

	days_ahead = [(date_temp - collectionTimeStamp).days for date_temp in dates]

	if(any(day_ahead >= 0 and day_ahead <= days_of_imagery_to_average for day_ahead in days_ahead)):
		dataset = xr.open_dataset(file_path, group='geophysical_data')
		nav_dataset = xr.open_dataset(file_path, group='navigation_data')
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

#features = np.zeros((florida_stats.shape[0], florida_stats.shape[1], 8, len(dates)))
#for i in range(florida_stats.shape[0]):
#	print('{}/{}'.format(i, florida_stats.shape[0]))
#	for j in range(florida_stats.shape[1]):
#		for k in range(len(dates)):
#			if(florida_stats[i, j, 1, k] == 0):
#				features[i, j, 0, k] = -1
#			else:
#				features[i, j, 0, k] = florida_stats[i, j, 0, k]/florida_stats[i, j, 1, k]

#			if(florida_stats[i, j, 3, k] == 0):
#				features[i, j, 1, k] = -1
#			else:
#				features[i, j, 1, k] = florida_stats[i, j, 2, k]/florida_stats[i, j, 3, k]

#			if(florida_stats[i, j, 5, k] == 0):
#				features[i, j, 2, k] = -1
#			else:
#				features[i, j, 2, k] = florida_stats[i, j, 4, k]/florida_stats[i, j, 5, k]

#			if(florida_stats[i, j, 7, k] == 0):
#				features[i, j, 3, k] = -1
#			else:
#				features[i, j, 3, k] = florida_stats[i, j, 6, k]/florida_stats[i, j, 7, k]

#			if(florida_stats[i, j, 9, k] == 0):
#				features[i, j, 4, k] = -1
#			else:
#				features[i, j, 4, k] = florida_stats[i, j, 8, k]/florida_stats[i, j, 9, k]

#			if(florida_stats[i, j, 11, k] == 0):
#				features[i, j, 5, k] = -1
#			else:
#				features[i, j, 5, k] = florida_stats[i, j, 10, k]/florida_stats[i, j, 11, k]

#			if(florida_stats[i, j, 13, k] == 0):
#				features[i, j, 6, k] = -1
#			else:
#				features[i, j, 6, k] = florida_stats[i, j, 12, k]/florida_stats[i, j, 13, k]

features = np.zeros((florida_stats.shape[0], florida_stats.shape[1], 8, len(dates)))
for k in range(len(dates)):
	print('{}/{}'.format(k, len(dates)))
	features[:, :, 0, k] = np.divide(florida_stats[:, :, 0, k], florida_stats[:, :, 1, k])
	inds = np.where(florida_stats[:, :, 1, k] == 0)
	features[inds[0], inds[1], 0, k] = -1

	features[:, :, 1, k] = np.divide(florida_stats[:, :, 2, k], florida_stats[:, :, 3, k])
	inds = np.where(florida_stats[:, :, 3, k] == 0)
	features[inds[0], inds[1], 1, k] = -1

	features[:, :, 2, k] = np.divide(florida_stats[:, :, 4, k], florida_stats[:, :, 5, k])
	inds = np.where(florida_stats[:, :, 5, k] == 0)
	features[inds[0], inds[1], 2, k] = -1

	features[:, :, 3, k] = np.divide(florida_stats[:, :, 6, k], florida_stats[:, :, 7, k])
	inds = np.where(florida_stats[:, :, 7, k] == 0)
	features[inds[0], inds[1], 3, k] = -1

	features[:, :, 4, k] = np.divide(florida_stats[:, :, 8, k], florida_stats[:, :, 9, k])
	inds = np.where(florida_stats[:, :, 9, k] == 0)
	features[inds[0], inds[1], 4, k] = -1

	features[:, :, 5, k] = np.divide(florida_stats[:, :, 10, k], florida_stats[:, :, 11, k])
	inds = np.where(florida_stats[:, :, 11, k] == 0)
	features[inds[0], inds[1], 5, k] = -1

	features[:, :, 6, k] = np.divide(florida_stats[:, :, 12, k], florida_stats[:, :, 13, k])
	inds = np.where(florida_stats[:, :, 13, k] == 0)
	features[inds[0], inds[1], 6, k] = -1

bedrock_x = np.load('florida_x.npy')
bedrock_y = np.load('florida_y.npy')
bedrock_z = np.load('florida_z.npy')

#Reduce data to only Southwest Florida
original_size = features.shape
florida_lats = np.reshape(florida_lats, (features.shape[0], features.shape[1]), order='C')
florida_lons = np.reshape(florida_lons, (features.shape[0], features.shape[1]), order='C')
florida_lats = florida_lats[minLonIdx:maxLonIdx,minLatIdx:maxLatIdx]
florida_lons = florida_lons[minLonIdx:maxLonIdx,minLatIdx:maxLatIdx]

np.save('latitudes.npy', florida_lats)
np.save('longitudes.npy', florida_lons)

florida_lats = np.reshape(florida_lats, (florida_lats.shape[0]*florida_lats.shape[1]))
florida_lons = np.reshape(florida_lons, (florida_lons.shape[0]*florida_lons.shape[1]))
features = features[minLonIdx:maxLonIdx, minLatIdx:maxLatIdx, :, :]
original_size = features.shape

features = np.reshape(features, (features.shape[0]*features.shape[1], 8, len(dates)))

orig_indices_bedrock = findMatrixCoordsBedrock(bedrock_x, bedrock_y, florida_lons, florida_lats)

for i in range(len(orig_indices_bedrock)):
	for j in range(len(dates)):
		features[i, 7, j] = bedrock_z[orig_indices_bedrock[i][0]][orig_indices_bedrock[i][1]]

land_mask = features[:,7,0] >= 0
land_mask = np.reshape(land_mask, (original_size[0], original_size[1]))

day_counter = 0
for day in dates:
	print('Processing days: {}/{}'.format(day_counter, len(dates)))

	day_features = np.squeeze(features[:, :, day_counter])
	#np.save('chlor_maps/day_features{}.npy'.format(day_counter), day_features)

	day_features_mask = (day_features[:,0] == -1) | (day_features[:,1] == -1) | (day_features[:,2] == -1) | (day_features[:,3] == -1) | (day_features[:,4] == -1) | (day_features[:,5] == -1) | (day_features[:,6] == -1)
	day_features_mask = np.reshape(day_features_mask, (original_size[0], original_size[1]), order='C')

	features_to_use = ['par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh']
	day_features = convertFeaturesByDepthPixelwise(day_features, features_to_use)

	featureTensor = torch.tensor(day_features).float().cuda()
	#day_features = np.reshape(day_features, (original_size[0], original_size[1], 6), order='C')

	configfilename = 'pixelwise+knn+dn'

	config = ConfigParser()
	config.read('configfiles/'+configfilename+'.ini')

	num_classes = config.getint('main', 'num_classes')
	randomseeds = json.loads(config.get('main', 'randomseeds'))
	use_nn_feature = config.getint('main', 'use_nn_feature')

	# 0 = No nn features, 1 = nn, 2 = weighted knn
	if(use_nn_feature == 1 or use_nn_feature == 2 or use_nn_feature == 3):
		file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

		df = pd.read_excel(file_path, engine='openpyxl')
		df_dates = df['Sample Date']
		df_lats = df['Latitude'].to_numpy()
		df_lons = df['Longitude'].to_numpy()
		df_concs = df['Karenia brevis abundance (cells/L)'].to_numpy()

		df_concs_log = np.log10(df_concs)/np.max(np.log10(df_concs))
		df_concs_log[np.isinf(df_concs_log)] = 0

		knn_features = np.zeros((featureTensor.shape[0], 1))

		searchdate = day
		weekbefore = searchdate - dt.timedelta(days=3)
		twoweeksbefore = searchdate - dt.timedelta(days=10)
		mask = (df_dates > twoweeksbefore) & (df_dates <= weekbefore)
		week_prior_inds = df_dates[mask].index.values

		beta = 1

		if(week_prior_inds.size):
			#Do some nearest neighbor thing with the last week's samples
			for i in range(len(knn_features)):
				physicalDistance = 100*np.sqrt((df_lats[week_prior_inds]-florida_lats[i])**2 + (df_lons[week_prior_inds]-florida_lons[i])**2)
				daysBack = (searchdate - df_dates[week_prior_inds]).astype('timedelta64[D]').values
				totalDistance = physicalDistance + beta*daysBack
				inverseDistance = 1/totalDistance
				NN_weights = inverseDistance/np.sum(inverseDistance)

				knn_features[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])

		knn_features_map = np.reshape(knn_features, (original_size[0], original_size[1]), order='C')

		featureTensorOriginal = featureTensor.clone()
		featureTensor = torch.cat((featureTensor, torch.tensor(knn_features).float().cuda()), dim=1)

	red_tide_output_sum_original = np.zeros((original_size[0], original_size[1], num_classes))
	red_tide_output_sum = np.zeros((original_size[0], original_size[1], num_classes))

	for model_number in range(len(randomseeds)):
		predictor = Predictor(featureTensor.shape[1], num_classes).cuda()
		predictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/{}model_correct.pt'.format(model_number)))
		predictor.eval()

		output = predictor(featureTensor)

		red_tide = output.detach().cpu().numpy()
		red_tide = np.reshape(red_tide, (original_size[0], original_size[1], num_classes), order='C')

		red_tide[day_features_mask] = -0.5
		red_tide[land_mask] = -1

		red_tide_output_sum = red_tide_output_sum + red_tide

		if(use_nn_feature == 3):
			originalPredictor = Predictor(featureTensorOriginal.shape[1], num_classes).cuda()
			originalPredictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/originalPredictor{}.pt'.format(model_number)))
			originalPredictor.eval()

			output = originalPredictor(featureTensorOriginal)

			red_tide = output[:, 1].detach().cpu().numpy()
			red_tide = np.reshape(red_tide, (original_size[0], original_size[1]), order='C')

			red_tide[day_features_mask] = -0.5
			red_tide[land_mask] = -1

			red_tide_output_sum_original = red_tide_output_sum_original + red_tide

	#red_tide_output_original = red_tide_output_sum_original/len(randomseeds)
	red_tide_output = red_tide_output_sum/len(randomseeds)

	#chlor_a_map = np.reshape(np.squeeze(day_features[:, 2]), (original_size[0], original_size[1]), order='C')
	#chlor_a_map[land_mask] = -1
	#chlor_a_map[day_features_mask] = -1

	#np.save('chlor_maps/chlor_image{}.npy'.format(day_counter), chlor_a_map)

	#plt.figure(dpi=500)
	#plt.imshow(chlor_a_map.T)
	#plt.colorbar()
	#plt.title('Chlor-a Concentration {}/{}/{}'.format(day.month, day.day, day.year))
	#plt.gca().invert_yaxis()
	#plt.savefig('chlor_maps/chlor_image{}.png'.format(str(day_counter).zfill(5)), bbox_inches='tight')

	np.save(save_folder+'red_tide_output{}.npy'.format(day_counter), red_tide_output)

	predicted_classes = np.argmax(red_tide_output, axis=2)
	predicted_classes = predicted_classes.astype(np.float32)

	plt.figure(dpi=500)
	plt.imshow(predicted_classes.T)
	plt.colorbar()
	plt.title('Red Tide Prediction {}/{}/{}'.format(day.month, day.day, day.year))
	plt.gca().invert_yaxis()
	plt.savefig(save_folder+'red_tide_image{}.png'.format(str(day_counter).zfill(5)), bbox_inches='tight')

	plt.close('all')

	day_counter = day_counter + 1
