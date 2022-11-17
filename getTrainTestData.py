import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from convertFeaturesByDepth import *

def getTrainTestData(data_folder, features_to_use, use_nn_feature, depth_normalize, randomseed, use_binned_data, pixelwise, test_only=False):
	np.random.seed(randomseed)

	positive_rawdata = np.load(data_folder+'positive_data.npy')
	positive_concs = np.load(data_folder+'positive_concs.npy')
	positive_dates = np.load(data_folder+'positive_dates.npy', allow_pickle=True)
	negative_rawdata = np.load(data_folder+'negative_data.npy')
	negative_concs = np.load(data_folder+'negative_concs.npy')
	negative_dates = np.load(data_folder+'negative_dates.npy', allow_pickle=True)

	feature_order = ['aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531',\
					 'Rrs_547', 'Rrs_555', 'Rrs_645', 'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx',\
					 'Kd_490', 'pic', 'poc', 'ipar', 'nflh', 'par', 'lon', 'lat', 'depth']

	feature_bands = []
	for i in range(len(features_to_use)):
		feature_bands.append(feature_order.index(features_to_use[i]))

	if(pixelwise == 0):
		positive_data = positive_rawdata[:, :, :, feature_bands]
		positive_lons = np.squeeze(positive_rawdata[:, 50, 50, 7])
		positive_lats = np.squeeze(positive_rawdata[:, 50, 50, 8])
		negative_data = negative_rawdata[:, :, :, feature_bands]
		negative_lons = np.squeeze(negative_rawdata[:, 50, 50, 7])
		negative_lats = np.squeeze(negative_rawdata[:, 50, 50, 8])

		positive_data = np.nan_to_num(positive_data)
		positive_data = np.moveaxis(positive_data, 3, 1)
		negative_data = np.nan_to_num(negative_data)
		negative_data = np.moveaxis(negative_data, 3, 1)

		data = np.concatenate((positive_data, negative_data), axis=0)
		concs = np.concatenate((positive_concs, negative_concs))
		lons = np.concatenate((positive_lons, negative_lons))
		lats = np.concatenate((positive_lats, negative_lats))
		dates = np.concatenate((positive_dates, negative_dates))
	else:
		positive_data = np.squeeze(positive_rawdata[:, 50, 50, feature_bands])
		positive_lons = np.squeeze(positive_rawdata[:, 50, 50, 7])
		positive_lats = np.squeeze(positive_rawdata[:, 50, 50, 8])
		negative_data = np.squeeze(negative_rawdata[:, 50, 50, feature_bands])
		negative_lons = np.squeeze(negative_rawdata[:, 50, 50, 7])
		negative_lats = np.squeeze(negative_rawdata[:, 50, 50, 8])

		positive_data = np.nan_to_num(positive_data)
		negative_data = np.nan_to_num(negative_data)

		data = np.concatenate((positive_data, negative_data), axis=0)
		concs = np.concatenate((positive_concs, negative_concs))
		lons = np.concatenate((positive_lons, negative_lons))
		lats = np.concatenate((positive_lats, negative_lats))
		dates = np.concatenate((positive_dates, negative_dates))

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

		testData = image_data[testInds, :, :, :]
		testKNN = knn_concs[testInds]
		testTargets = targets[testInds]
		testDates = dates[testInds]
	else:
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

		trainData = data[trainInds, :]
		trainKNN = knn_concs[trainInds]
		trainTargets = targets[trainInds]
		trainDates = dates[trainInds]

		testData = data[testInds, :]
		testKNN = knn_concs[testInds]
		testTargets = targets[testInds]
		testDates = dates[testInds]

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

	return trainData, trainKNN, trainTargets, trainDates, testData, testKNN, testTargets, testDates