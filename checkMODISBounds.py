import numpy as np

# Check if a value is within valid range for that feature according to https://oceancolor.gsfc.nasa.gov/docs/format/l2oc_modis/
# data - array
# feature - string of feature name
# if data is outside of range, set it to nan
def checkMODISBounds(data, feature):
	if(feature == 'aot_869'):
		# Valid range: (0.000, 3.000)
		data[(data < 0.000) | (data > 3.000)] = np.nan
	elif(feature == 'angstrom'):
		# Valid range: (-0.500, 3.000)
		data[(data < -0.500) | (data > 3.000)] = np.nan
	elif(feature == 'Rrs_412'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_443'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_469'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_488'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_531'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_547'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_555'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_645'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_667'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'Rrs_678'):
		# Valid range: (-0.010, 0.100)
		data[(data < -0.010) | (data > 0.100)] = np.nan
	elif(feature == 'chlor_a'):
		# Valid range: (0.001, 100)
		data[(data < 0.001) | (data > 100)] = np.nan
	elif(feature == 'chl_ocx'):
		# Valid range: (0.001, 100)
		data[(data < 0.001) | (data > 100)] = np.nan
	elif(feature == 'Kd_490'):
		# Valid range: (0.010, 6.000)
		data[(data < 0.010) | (data > 6.000)] = np.nan
	elif(feature == 'pic'):
		# Valid range: (0.000, 0.125)
		data[(data < 0.000) | (data > 0.125)] = np.nan
	elif(feature == 'poc'):
		# Valid range: (0.000, 1000.000)
		data[(data < 0.000) | (data > 1000.000)] = np.nan
	elif(feature == 'ipar'):
		# Valid range: None given
		pass
	elif(feature == 'nflh'):
		# Valid range: (-0.500, 5.000)
		data[(data < -0.500) | (data > 5.000)] = np.nan
	elif(feature == 'par'):
		# Valid range: (0.000, 18.928)
		data[(data < 0.000) | (data > 18.928)] = np.nan
	
	else:
		print('Feature name invalid!')

	return data