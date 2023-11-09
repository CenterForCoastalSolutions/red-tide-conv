import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from convertFeaturesByDepth import *

def getTrainTestData(data_folder, features_to_use, use_nn_feature, depth_normalize, randomseed, use_binned_data, pixelwise, test_only=False):
	np.random.seed(randomseed)

	background_rawdata = np.load(data_folder+'background_data.npy')
	background_concs = np.load(data_folder+'background_concs.npy')
	background_dates = np.load(data_folder+'background_dates.npy', allow_pickle=True)
	verylow_rawdata = np.load(data_folder+'verylow_data.npy')
	verylow_concs = np.load(data_folder+'verylow_concs.npy')
	verylow_dates = np.load(data_folder+'verylow_dates.npy', allow_pickle=True)
	lowbelow50000_rawdata = np.load(data_folder+'lowbelow50000_data.npy')
	lowbelow50000_concs = np.load(data_folder+'lowbelow50000_concs.npy')
	lowbelow50000_dates = np.load(data_folder+'lowbelow50000_dates.npy', allow_pickle=True)
	lowabove50000_rawdata = np.load(data_folder+'lowabove50000_data.npy')
	lowabove50000_concs = np.load(data_folder+'lowabove50000_concs.npy')
	lowabove50000_dates = np.load(data_folder+'lowabove50000_dates.npy', allow_pickle=True)
	medium_rawdata = np.load(data_folder+'medium_data.npy')
	medium_concs = np.load(data_folder+'medium_concs.npy')
	medium_dates = np.load(data_folder+'medium_dates.npy', allow_pickle=True)
	high_rawdata = np.load(data_folder+'high_data.npy')
	high_concs = np.load(data_folder+'high_concs.npy')
	high_dates = np.load(data_folder+'high_dates.npy', allow_pickle=True)

	feature_order = ['aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531',\
					 'Rrs_547', 'Rrs_555', 'Rrs_645', 'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx',\
					 'Kd_490', 'pic', 'poc', 'ipar', 'nflh', 'par', 'lon', 'lat', 'depth']

	feature_bands = []
	for i in range(len(features_to_use)):
		feature_bands.append(feature_order.index(features_to_use[i]))

	if(pixelwise == 0):
		background_data = np.squeeze(background_rawdata[:, :, :, feature_bands])
		background_lons = np.squeeze(background_rawdata[:, 50, 50, 7])
		background_lats = np.squeeze(background_rawdata[:, 50, 50, 8])
		verylow_data = np.squeeze(verylow_rawdata[:, :, :, feature_bands])
		verylow_lons = np.squeeze(verylow_rawdata[:, 50, 50, 7])
		verylow_lats = np.squeeze(verylow_rawdata[:, 50, 50, 8])
		lowbelow50000_data = np.squeeze(lowbelow50000_rawdata[:, :, :, feature_bands])
		lowbelow50000_lons = np.squeeze(lowbelow50000_rawdata[:, 50, 50, 7])
		lowbelow50000_lats = np.squeeze(lowbelow50000_rawdata[:, 50, 50, 8])
		lowabove50000_data = np.squeeze(lowabove50000_rawdata[:, :, :, feature_bands])
		lowabove50000_lons = np.squeeze(lowabove50000_rawdata[:, 50, 50, 7])
		lowabove50000_lats = np.squeeze(lowabove50000_rawdata[:, 50, 50, 8])
		medium_data = np.squeeze(medium_rawdata[:, :, :, feature_bands])
		medium_lons = np.squeeze(medium_rawdata[:, 50, 50, 7])
		medium_lats = np.squeeze(medium_rawdata[:, 50, 50, 8])
		high_data = np.squeeze(high_rawdata[:, :, :, feature_bands])
		high_lons = np.squeeze(high_rawdata[:, 50, 50, 7])
		high_lats = np.squeeze(high_rawdata[:, 50, 50, 8])

		background_data = np.nan_to_num(background_data)
		background_data = np.moveaxis(background_data, 3, 1)
		verylow_data = np.nan_to_num(verylow_data)
		verylow_data = np.moveaxis(verylow_data, 3, 1)
		lowbelow50000_data = np.nan_to_num(lowbelow50000_data)
		lowbelow50000_data = np.moveaxis(lowbelow50000_data, 3, 1)
		lowabove50000_data = np.nan_to_num(lowabove50000_data)
		lowabove50000_data = np.moveaxis(lowabove50000_data, 3, 1)
		medium_data = np.nan_to_num(medium_data)
		medium_data = np.moveaxis(medium_data, 3, 1)
		high_data = np.nan_to_num(high_data)
		high_data = np.moveaxis(high_data, 3, 1)

		data = np.concatenate((background_data, verylow_data, lowbelow50000_data, lowabove50000_data, medium_data, high_data), axis=0)
		concs = np.concatenate((background_concs, verylow_concs, lowbelow50000_concs, lowabove50000_concs, medium_concs, high_concs))
		lons = np.concatenate((background_lons, verylow_lons, lowbelow50000_lons, lowabove50000_lons, medium_lons, high_lons))
		lats = np.concatenate((background_lats, verylow_lats, lowbelow50000_lats, lowabove50000_lats, medium_lats, high_lats))
		dates = np.concatenate((background_dates, verylow_dates, lowbelow50000_dates, lowabove50000_dates, medium_dates, high_dates))
	else:
		background_data = np.squeeze(background_rawdata[:, 50, 50, feature_bands])
		background_lons = np.squeeze(background_rawdata[:, 50, 50, 7])
		background_lats = np.squeeze(background_rawdata[:, 50, 50, 8])
		verylow_data = np.squeeze(verylow_rawdata[:, 50, 50, feature_bands])
		verylow_lons = np.squeeze(verylow_rawdata[:, 50, 50, 7])
		verylow_lats = np.squeeze(verylow_rawdata[:, 50, 50, 8])
		lowbelow50000_data = np.squeeze(lowbelow50000_rawdata[:, 50, 50, feature_bands])
		lowbelow50000_lons = np.squeeze(lowbelow50000_rawdata[:, 50, 50, 7])
		lowbelow50000_lats = np.squeeze(lowbelow50000_rawdata[:, 50, 50, 8])
		lowabove50000_data = np.squeeze(lowabove50000_rawdata[:, 50, 50, feature_bands])
		lowabove50000_lons = np.squeeze(lowabove50000_rawdata[:, 50, 50, 7])
		lowabove50000_lats = np.squeeze(lowabove50000_rawdata[:, 50, 50, 8])
		medium_data = np.squeeze(medium_rawdata[:, 50, 50, feature_bands])
		medium_lons = np.squeeze(medium_rawdata[:, 50, 50, 7])
		medium_lats = np.squeeze(medium_rawdata[:, 50, 50, 8])
		high_data = np.squeeze(high_rawdata[:, 50, 50, feature_bands])
		high_lons = np.squeeze(high_rawdata[:, 50, 50, 7])
		high_lats = np.squeeze(high_rawdata[:, 50, 50, 8])

		background_data = np.nan_to_num(background_data)
		verylow_data = np.nan_to_num(verylow_data)
		lowbelow50000_data = np.nan_to_num(lowbelow50000_data)
		lowabove50000_data = np.nan_to_num(lowabove50000_data)
		medium_data = np.nan_to_num(medium_data)
		high_data = np.nan_to_num(high_data)

		data = np.concatenate((background_data, verylow_data, lowbelow50000_data, lowabove50000_data, medium_data, high_data), axis=0)
		concs = np.concatenate((background_concs, verylow_concs, lowbelow50000_concs, lowabove50000_concs, medium_concs, high_concs))
		lons = np.concatenate((background_lons, verylow_lons, lowbelow50000_lons, lowabove50000_lons, medium_lons, high_lons))
		lats = np.concatenate((background_lats, verylow_lats, lowbelow50000_lats, lowabove50000_lats, medium_lats, high_lats))
		dates = np.concatenate((background_dates, verylow_dates, lowbelow50000_dates, lowabove50000_dates, medium_dates, high_dates))

	beta = 1

	# 0 = No nn features, 1 = weighted knn
	if(use_nn_feature == 1):
		file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

		df = pd.read_excel(file_path, engine='openpyxl')
		df_dates = df['Sample Date']
		df_lats = df['Latitude'].to_numpy()
		df_lons = df['Longitude'].to_numpy()
		df_concs = df['Karenia brevis abundance (cells/L)'].to_numpy()

		if(use_binned_data == 1):
			background_inds = np.where(df_concs < 1000)[0]
			verylow_inds = np.where((df_concs >= 1000) & (df_concs < 10000))[0]
			low_inds = np.where((df_concs >= 10000) & (df_concs < 100000))[0]
			medium_inds = np.where((df_concs >= 100000) & (df_concs < 1000000))[0]
			high_inds = np.where(df_concs >= 1000000)[0]
			
			df_concs[background_inds] = 0
			df_concs[verylow_inds] = 1000
			df_concs[low_inds] = 10000
			df_concs[medium_inds] = 100000
			df_concs[high_inds] = 1000000

		df_concs_log = np.log10(df_concs)/np.max(np.log10(df_concs))
		df_concs_log[np.isinf(df_concs_log)] = 0

		#dataDates = pd.DatetimeIndex(dates)
		dataDates = dates
		knn_concs = np.zeros((data.shape[0]))
		#Do some nearest neighbor thing with the last week's samples
		for i in range(len(dataDates)):
			searchdate = dataDates[i]
			weekbefore = searchdate - dt.timedelta(days=3)
			twoweeksbefore = searchdate - dt.timedelta(days=10)
			mask = (df_dates > twoweeksbefore) & (df_dates <= weekbefore)
			week_prior_inds = df_dates[mask].index.values

			if(week_prior_inds.size):
				physicalDistance = np.zeros((len(week_prior_inds), 1))
				for j in range(len(week_prior_inds)):
					oldLocation = (df_lats[week_prior_inds][j], df_lons[week_prior_inds][j])
					currLocation = (lats[i], lons[i])
					physicalDistance[j] = geodesic(oldLocation, currLocation).km
				daysBack = (searchdate - df_dates[week_prior_inds]).astype('timedelta64[D]').values
				totalDistance = physicalDistance + beta*daysBack
				inverseDistance = 1/totalDistance
				NN_weights = inverseDistance/np.sum(inverseDistance)

				knn_concs[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])
	else:
		knn_concs = np.zeros((data.shape[0]))

	years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,\
			 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
	testYears = np.random.choice(years, 3, replace=False)
	trainYears = list(set(years)-set(testYears))

	trainInds = []
	testInds = []

	for i in range(len(dates)):
		date = dates[i][()]
		if(date.year in trainYears):
			trainInds.append(i)
		if(date.year in testYears):
			testInds.append(i)

	trainInds = np.array(trainInds)
	testInds = np.array(testInds)

	if(pixelwise == 0):
		image_data = np.zeros((data.shape[0], data.shape[1], 224, 224))
		image_data[:, :, 62:163, 62:163] = data

		classes = np.zeros((concs.shape[0], 1))

		for i in range(len(classes)):
			if(concs[i] < 50000):
				classes[i] = 0
			else:
				classes[i] = 1

		classes = classes.astype(int)

		targets = np.zeros((classes.shape[0], 2))
		for i in range(len(classes)):
			targets[i, classes[i]] = 1

		trainData = image_data[trainInds, :, :, :]
		trainKNN = knn_concs[trainInds]
		trainTargets = targets[trainInds]
		trainDates = dates[trainInds]
		trainConcs = concs[trainInds]

		testData = image_data[testInds, :, :, :]
		testKNN = knn_concs[testInds]
		testTargets = targets[testInds]
		testDates = dates[testInds]
		testConcs = concs[testInds]
	else:
		classes = np.zeros((concs.shape[0], 1))

		for i in range(len(classes)):
			if(concs[i] < 1000):
				classes[i] = 0
			elif(concs[i] >= 1000 and concs[i] < 10000):
				classes[i] = 1
			elif(concs[i] >= 10000 and concs[i] < 50000):
				classes[i] = 2
			elif(concs[i] >= 50000 and concs[i] < 100000):
				classes[i] = 3
			elif(concs[i] >= 100000 and concs[i] < 1000000):
				classes[i] = 4
			else:
				classes[i] = 5

		classes = classes.astype(int)

		targets = np.zeros((classes.shape[0], 6))
		for i in range(len(classes)):
			targets[i, classes[i]] = 1

		trainData = data[trainInds, :]
		trainKNN = knn_concs[trainInds]
		trainTargets = targets[trainInds]
		trainDates = dates[trainInds]
		trainConcs = concs[trainInds]

		testData = data[testInds, :]
		testKNN = knn_concs[testInds]
		testTargets = targets[testInds]
		testDates = dates[testInds]
		testConcs = concs[testInds]

	if(depth_normalize == 1):
		if(pixelwise == 0):
			if(test_only==False):
				trainData = convertFeaturesByDepth(trainData, features_to_use[0:-1])
				testData = convertFeaturesByDepth(testData, features_to_use[0:-1])
			else:
				testData = convertFeaturesByDepth(testData, features_to_use[0:-1])
		else:
			trainData = convertFeaturesByDepthPixelwise(trainData, features_to_use[0:-1])
			testData = convertFeaturesByDepthPixelwise(testData, features_to_use[0:-1])

	return trainData, trainKNN, trainTargets, trainDates, trainConcs, testData, testKNN, testTargets, testDates, testConcs