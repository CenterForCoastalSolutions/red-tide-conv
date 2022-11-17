# This function assesses the red tide images based on: if the center pixel is available, the number of pixels available,
# the number of water pixels total
#
# Returns the images with the highest percentage of valid water pixels


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def pullConvData(numPositives, numNegatives):
	save_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/'

	# data is (points, longitude, latitude, features)
	data = np.load(save_folder+'red_tide_conv_images_correct.npy')
	concs = np.load(save_folder+'red_tide_conv_concs_correct.npy')
	dates = np.load(save_folder+'red_tide_conv_dates_correct.npy', allow_pickle=True)

	assessment = np.zeros((data.shape[0], 3))

	for i in range(data.shape[0]):
		if(np.isnan(data[i, 50, 50, 2]) == False):
			assessment[i, 0] = 1
		assessment[i, 1] = (101*101) - np.sum(np.isnan(data[i, :, :, 2]))
		assessment[i, 2] = np.sum(data[i, :, :, 9] < 0)

	valid_middle_inds = assessment[:, 0] == 1
	valid_middle_assessment = assessment[valid_middle_inds, :]
	valid_middle_data = data[valid_middle_inds, :, :, :]
	valid_middle_concs = concs[valid_middle_inds]
	valid_middle_dates = dates[valid_middle_inds]

	del data

	valid_middle_positives_assessment = valid_middle_assessment[valid_middle_concs > 50000, :]
	valid_middle_negatives_assessment = valid_middle_assessment[valid_middle_concs == 0, :]
	#valid_middle_negatives_assessment = valid_middle_assessment[valid_middle_concs < 50000, :]

	valid_middle_positives_data = valid_middle_data[valid_middle_concs > 50000, :, :, :]
	valid_middle_negatives_data = valid_middle_data[valid_middle_concs == 0, :, :, :]
	#valid_middle_negatives_data = valid_middle_data[valid_middle_concs < 50000, :, :, :]

	valid_middle_positives_concs = valid_middle_concs[valid_middle_concs > 50000]
	valid_middle_negatives_concs = valid_middle_concs[valid_middle_concs == 0]
	#valid_middle_negatives_concs = valid_middle_concs[valid_middle_concs < 50000]

	valid_middle_positives_dates = valid_middle_dates[valid_middle_concs > 50000]
	valid_middle_negatives_dates = valid_middle_dates[valid_middle_concs == 0]
	#valid_middle_negatives_dates = valid_middle_dates[valid_middle_concs < 50000]

	#negatives_sort_array = valid_middle_negatives_assessment[:, 1]/valid_middle_negatives_assessment[:, 2]
	#positives_sort_array = valid_middle_positives_assessment[:, 1]/valid_middle_positives_assessment[:, 2]
	negatives_sort_array = valid_middle_negatives_assessment[:, 1]
	positives_sort_array = valid_middle_positives_assessment[:, 1]

	valid_negatives = np.where(negatives_sort_array > 5100)[0]
	valid_positives = np.where(positives_sort_array > 5100)[0]

	#best_negatives = np.argsort(negatives_sort_array)
	#best_positives = np.argsort(positives_sort_array)

	if(numPositives > len(valid_positives) or numNegatives > len(valid_negatives)):
		raise ValueError('Not enough samples to satisfy data requested')
	else:
		positive_inds = np.random.choice(valid_positives, numPositives, replace=False)
		negative_inds = np.random.choice(valid_negatives, numNegatives, replace=False)

		positive_data = valid_middle_positives_data[positive_inds, :, :, :]
		negative_data = valid_middle_negatives_data[negative_inds, :, :, :]

		print(np.mean(positive_data[:, 50, 50, 20]))
		print(np.mean(negative_data[:, 50, 50, 20]))

		map = Basemap(llcrnrlon=-84, llcrnrlat=24, urcrnrlon=-79, urcrnrlat=29, projection='lcc', lon_0=-82, lat_0=26.5)
		map.drawmapboundary(fill_color='aqua')
		map.fillcontinents(color='#cc9955', lake_color='aqua', zorder=0)
		map.drawcoastlines(color='0.15')
		positive_lons = positive_data[:, 50, 50, 20]
		positive_lats = positive_data[:, 50, 50, 21]
		x, y = map(positive_lons, positive_lats)
		map.scatter(x, y, marker='D', color='m')
		negative_lons = negative_data[:, 50, 50, 20]
		negative_lats = negative_data[:, 50, 50, 21]
		x2, y2 = map(negative_lons, negative_lats)
		map.scatter(x2, y2, marker='D', color='k')
		plt.savefig('test.png')

		positive_concs = valid_middle_positives_concs[positive_inds]
		negative_concs = valid_middle_negatives_concs[negative_inds]

		positive_dates = valid_middle_positives_dates[positive_inds]
		negative_dates = valid_middle_negatives_dates[negative_inds]

		return positive_data, positive_concs, positive_dates, negative_data, negative_concs, negative_dates