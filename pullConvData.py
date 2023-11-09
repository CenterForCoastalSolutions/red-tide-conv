# This function assesses the red tide images based on: if the center pixel is available, the number of pixels available,
# the number of water pixels total
#
# Returns the images with the highest percentage of valid water pixels


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def pullConvData(numBackground, numVerylow, numLowBelow50000, numLowAbove50000, numMedium, numHigh):
	save_folder = '/data-drive-12521/MODIS-OC/MODIS-OC-data/'

	# data is (points, longitude, latitude, features)
	data = np.load(save_folder+'red_tide_conv_images_correct.npy')
	concs = np.load(save_folder+'red_tide_conv_concs_correct.npy')
	dates = np.load(save_folder+'red_tide_conv_dates_correct.npy', allow_pickle=True)

	assessment = np.zeros((data.shape[0], 3))

	for i in range(data.shape[0]):
		print('processing {}/{}'.format(i, data.shape[0]))
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

	print(valid_middle_assessment[valid_middle_concs < 1000].shape)
	print(valid_middle_assessment[(valid_middle_concs >= 1000) & (valid_middle_concs < 10000)].shape)
	print(valid_middle_assessment[(valid_middle_concs >= 10000) & (valid_middle_concs < 50000)].shape)
	print(valid_middle_assessment[(valid_middle_concs >= 50000) & (valid_middle_concs < 100000)].shape)
	print(valid_middle_assessment[(valid_middle_concs >= 100000) & (valid_middle_concs < 1000000)].shape)
	print(valid_middle_assessment[(valid_middle_concs >= 1000000)].shape)

	#valid_middle_positives_assessment = valid_middle_assessment[valid_middle_concs > 50000, :]
	#valid_middle_negatives_assessment = valid_middle_assessment[valid_middle_concs == 0, :]
	#valid_middle_negatives_assessment = valid_middle_assessment[valid_middle_concs < 50000, :]

	#valid_middle_positives_data = valid_middle_data[valid_middle_concs > 50000, :, :, :]
	#valid_middle_negatives_data = valid_middle_data[valid_middle_concs == 0, :, :, :]
	#valid_middle_negatives_data = valid_middle_data[valid_middle_concs < 50000, :, :, :]

	valid_middle_background_data = valid_middle_data[valid_middle_concs < 1000, :, :, :]
	valid_middle_verylow_data = valid_middle_data[(valid_middle_concs >= 1000) & (valid_middle_concs < 10000), :, :, :]
	valid_middle_lowbelow50000_data = valid_middle_data[(valid_middle_concs >= 10000) & (valid_middle_concs < 50000), :, :, :]
	valid_middle_lowabove50000_data = valid_middle_data[(valid_middle_concs >= 50000) & (valid_middle_concs < 100000), :, :, :]
	valid_middle_medium_data = valid_middle_data[(valid_middle_concs >= 100000) & (valid_middle_concs < 1000000), :, :, :]
	valid_middle_high_data = valid_middle_data[valid_middle_concs >= 1000000, :, :, :]

	#valid_middle_positives_concs = valid_middle_concs[valid_middle_concs > 50000]
	#valid_middle_negatives_concs = valid_middle_concs[valid_middle_concs == 0]
	#valid_middle_negatives_concs = valid_middle_concs[valid_middle_concs < 50000]

	valid_middle_background_concs = valid_middle_concs[valid_middle_concs < 1000]
	valid_middle_verylow_concs = valid_middle_concs[(valid_middle_concs >= 1000) & (valid_middle_concs < 10000)]
	valid_middle_lowbelow50000_concs = valid_middle_concs[(valid_middle_concs >= 10000) & (valid_middle_concs < 50000)]
	valid_middle_lowabove50000_concs = valid_middle_concs[(valid_middle_concs >= 50000) & (valid_middle_concs < 100000)]
	valid_middle_medium_concs = valid_middle_concs[(valid_middle_concs >= 100000) & (valid_middle_concs < 1000000)]
	valid_middle_high_concs = valid_middle_concs[valid_middle_concs >= 1000000]

	#valid_middle_positives_dates = valid_middle_dates[valid_middle_concs > 50000]
	#valid_middle_negatives_dates = valid_middle_dates[valid_middle_concs == 0]
	#valid_middle_negatives_dates = valid_middle_dates[valid_middle_concs < 50000]

	valid_middle_background_dates = valid_middle_dates[valid_middle_concs < 1000]
	valid_middle_verylow_dates = valid_middle_dates[(valid_middle_concs >= 1000) & (valid_middle_concs < 10000)]
	valid_middle_lowbelow50000_dates = valid_middle_dates[(valid_middle_concs >= 10000) & (valid_middle_concs < 50000)]
	valid_middle_lowabove50000_dates = valid_middle_dates[(valid_middle_concs >= 50000) & (valid_middle_concs < 100000)]
	valid_middle_medium_dates = valid_middle_dates[(valid_middle_concs >= 100000) & (valid_middle_concs < 1000000)]
	valid_middle_high_dates = valid_middle_dates[valid_middle_concs >= 1000000]

	#negatives_sort_array = valid_middle_negatives_assessment[:, 1]
	#positives_sort_array = valid_middle_positives_assessment[:, 1]

	#valid_negatives = np.where(negatives_sort_array > 5100)[0]
	#valid_positives = np.where(positives_sort_array > 5100)[0]

	#best_negatives = np.argsort(negatives_sort_array)
	#best_positives = np.argsort(positives_sort_array)

	#np.save('convDataPlotting/positive_locations.npy', valid_middle_positives_data[:, 50, 50, 20:22])
	#np.save('convDataPlotting/negative_locations.npy', valid_middle_negatives_data[:, 50, 50, 20:22])
	#np.save('convDataPlotting/positive_concs.npy', valid_middle_positives_concs)
	#np.save('convDataPlotting/negative_concs.npy', valid_middle_negatives_concs)

	valid_background = len(valid_middle_background_data)
	valid_verylow = len(valid_middle_verylow_data)
	valid_lowbelow50000 = len(valid_middle_lowbelow50000_data)
	valid_lowabove50000 = len(valid_middle_lowabove50000_data)
	valid_medium = len(valid_middle_medium_data)
	valid_high = len(valid_middle_high_data)

	if(numBackground > valid_background or numVerylow > valid_verylow or numLowBelow50000 > valid_lowbelow50000 or numLowAbove50000 > valid_lowabove50000 or numMedium > valid_medium or numHigh > valid_high):
		raise ValueError('Not enough samples to satisfy data requested')
	else:
		background_inds = np.random.choice(valid_background, numBackground, replace=False)
		verylow_inds = np.random.choice(valid_verylow, numVerylow, replace=False)
		lowbelow50000_inds = np.random.choice(valid_lowbelow50000, numLowBelow50000, replace=False)
		lowabove50000_inds = np.random.choice(valid_lowabove50000, numLowAbove50000, replace=False)
		medium_inds = np.random.choice(valid_medium, numMedium, replace=False)
		high_inds = np.random.choice(valid_high, numHigh, replace=False)

		background_data = valid_middle_background_data[background_inds, :, :, :]
		verylow_data = valid_middle_verylow_data[verylow_inds, :, :, :]
		lowbelow50000_data = valid_middle_lowbelow50000_data[lowbelow50000_inds, :, :, :]
		lowabove50000_data = valid_middle_lowabove50000_data[lowabove50000_inds, :, :, :]
		medium_data = valid_middle_medium_data[medium_inds, :, :, :]
		high_data = valid_middle_high_data[high_inds, :, :, :]

		#print(np.mean(positive_data[:, 50, 50, 20]))
		#print(np.mean(negative_data[:, 50, 50, 20]))

		#map = Basemap(llcrnrlon=-84, llcrnrlat=24, urcrnrlon=-79, urcrnrlat=29, projection='lcc', lon_0=-82, lat_0=26.5)
		#map.drawmapboundary(fill_color='aqua')
		#map.fillcontinents(color='#cc9955', lake_color='aqua', zorder=0)
		#map.drawcoastlines(color='0.15')
		#positive_lons = positive_data[:, 50, 50, 20]
		#positive_lats = positive_data[:, 50, 50, 21]
		#x, y = map(positive_lons, positive_lats)
		#map.scatter(x, y, marker='D', color='m')
		#negative_lons = negative_data[:, 50, 50, 20]
		#negative_lats = negative_data[:, 50, 50, 21]
		#x2, y2 = map(negative_lons, negative_lats)
		#map.scatter(x2, y2, marker='D', color='k')
		#plt.savefig('test.png')

		#positive_concs = valid_middle_positives_concs[positive_inds]
		#negative_concs = valid_middle_negatives_concs[negative_inds]

		background_concs = valid_middle_background_concs[background_inds]
		verylow_concs = valid_middle_verylow_concs[verylow_inds]
		lowbelow50000_concs = valid_middle_lowbelow50000_concs[lowbelow50000_inds]
		lowabove50000_concs = valid_middle_lowabove50000_concs[lowabove50000_inds]
		medium_concs = valid_middle_medium_concs[medium_inds]
		high_concs = valid_middle_high_concs[high_inds]

		#positive_dates = valid_middle_positives_dates[positive_inds]
		#negative_dates = valid_middle_negatives_dates[negative_inds]

		background_dates = valid_middle_background_dates[background_inds]
		verylow_dates = valid_middle_verylow_dates[verylow_inds]
		lowbelow50000_dates = valid_middle_lowbelow50000_dates[lowbelow50000_inds]
		lowabove50000_dates = valid_middle_lowabove50000_dates[lowabove50000_inds]
		medium_dates = valid_middle_medium_dates[medium_inds]
		high_dates = valid_middle_high_dates[high_inds]

		return background_data, background_concs, background_dates, verylow_data, verylow_concs, verylow_dates, lowbelow50000_data, lowbelow50000_concs, lowbelow50000_dates, lowabove50000_data, lowabove50000_concs, lowabove50000_dates, medium_data, medium_concs, medium_dates, high_data, high_concs, high_dates


if __name__ == "__main__":
	background_data, background_concs, background_dates, verylow_data, verylow_concs, verylow_dates, lowbelow50000_data, lowbelow50000_concs, lowbelow50000_dates, lowabove50000_data, lowabove50000_concs, lowabove50000_dates, medium_data, medium_concs, medium_dates, high_data, high_concs, high_dates = pullConvData(350, 350, 350, 350, 350, 350)