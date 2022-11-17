import os
import numpy as np

def ensure_folder(foldername):
	if not os.path.exists(foldername):
		os.makedirs(foldername)

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx