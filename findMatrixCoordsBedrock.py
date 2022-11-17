import xarray as xr
import netCDF4
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

# Finds the nearest matrix coordinates corresponding to the
# Lats and lons of those provided

def findMatrixCoordsBedrock(bedrock_lon, bedrock_lat, sample_lons, sample_lats):
	orig_indices = []
	for i in range(len(sample_lats)):
		lat_index = find_nearest(bedrock_lat, sample_lats[i])
		lon_index = find_nearest(bedrock_lon, sample_lons[i])
		orig_indices.append((lat_index, lon_index))

	return orig_indices