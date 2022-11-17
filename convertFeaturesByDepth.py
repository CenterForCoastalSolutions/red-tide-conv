import numpy as np
import time

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def convertFeaturesByDepth(features, features_to_use):
	#Find depths stats size
	angstrom_sums_keys = np.load('depth_stats/angstrom_sums_keys.npy')

	depth_stats = np.zeros((len(features_to_use), 3, len(angstrom_sums_keys)))
	for i in range(len(features_to_use)):
		depth_stats[i, 0, :] = np.load('depth_stats/'+features_to_use[i]+'_sums_keys.npy')
		depth_stats[i, 1, :] = np.load('depth_stats/'+features_to_use[i]+'_sums_means.npy')
		depth_stats[i, 2, :] = np.load('depth_stats/'+features_to_use[i]+'_sums_stds.npy')

	featuresDepthConverted = np.zeros_like(features)

	for i in range(features.shape[0]):
		if(i%100 == 0):
			print('Converting data: {}/{}'.format(i, features.shape[0]))
		# Depth is last feature
		depth = features[i, -1, :, :]
		closest_depth_idx = np.zeros_like(depth)
		for j in range(depth.shape[0]):
			for k in range(depth.shape[1]):
				closest_depth_idx[j, k] = find_nearest(depth_stats[0, 0, :], depth[j, k])
		for l in range(len(features_to_use)):
			for j in range(depth.shape[0]):
				for k in range(depth.shape[1]):
					mean = depth_stats[l, 1, int(closest_depth_idx[j, k])]
					std = depth_stats[l, 2, int(closest_depth_idx[j, k])]
					featuresDepthConverted[i, l, j, k] = (features[i, l, j, k]-mean)/std
		featuresDepthConverted[i, -1, :, :] = features[i, -1, :, :]

	return featuresDepthConverted

def convertFeaturesByDepthPixelwise(features, features_to_use):
	#Find depths stats size
	angstrom_sums_keys = np.load('depth_stats/angstrom_sums_keys.npy')

	depth_stats = np.zeros((len(features_to_use), 3, len(angstrom_sums_keys)))
	for i in range(len(features_to_use)):
		depth_stats[i, 0, :] = np.load('depth_stats/'+features_to_use[i]+'_sums_keys.npy')
		depth_stats[i, 1, :] = np.load('depth_stats/'+features_to_use[i]+'_sums_means.npy')
		depth_stats[i, 2, :] = np.load('depth_stats/'+features_to_use[i]+'_sums_stds.npy')

	featuresDepthConverted = np.zeros((features.shape[0], features.shape[1]-1))

	for i in range(features.shape[0]):
		# Depth is last feature
		depth = int(features[i,-1])
		closest_depth_idx = find_nearest(depth_stats[0, 0, :], depth)
		for j in range(featuresDepthConverted.shape[1]):
			mean = depth_stats[j, 1, closest_depth_idx]
			std = depth_stats[j, 2, closest_depth_idx]
			featuresDepthConverted[i, j] = (features[i, j]-mean)/std

	return featuresDepthConverted