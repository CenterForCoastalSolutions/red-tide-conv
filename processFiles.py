import xarray as xr
import netCDF4
import math
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import spatial
from checkMODISBounds import *
#from findMatrixCoords import *
from findMatrixCoordsBedrock import *
#from utils import *
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools

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


save_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/red_tide_conv_images_correct/'

file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')
df_dates = df['Sample Date'].tolist()
df_depths = df['Sample Depth (m)'].tolist()
df_lats = df['Latitude'].tolist()
df_lons = df['Longitude'].tolist()
df_concs = df['Karenia brevis abundance (cells/L)']

florida_lon = np.load('florida_x.npy')
florida_lat = np.load('florida_y.npy')
florida_depth = np.load('florida_z.npy')

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

data_dates = []

for i in range(len(data_list)):
	if(i%100 == 0):
		print('Getting file dates: {}/{}'.format(i, len(data_list)))
	file_id = data_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDate = fh.time_coverage_start[0:10]
	collectionTimeStamp = pd.Timestamp(int(collectionDate[0:4]), int(collectionDate[5:7]), int(collectionDate[8:10]), 0)
	data_dates.append(collectionTimeStamp)


for i in range(len(df_dates)):
	print('Processing sample {}/{}'.format(i, len(df_dates)))
	sample_data_inds = [j for j in range(len(data_dates)) if (df_dates[i] - data_dates[j]).days <= 10 and (df_dates[i] - data_dates[j]).days >= 0]

	if(len(sample_data_inds) == 0):
		continue
	else:
		sample_lat = df_lats[i]
		sample_lon = df_lons[i]
		#100 km x 100 km grid
		lon_vals = np.linspace(sample_lon-0.45, sample_lon+0.45, 101)
		lat_vals = np.linspace(sample_lat-0.45, sample_lat+0.45, 101)
		xx, yy = np.meshgrid(lon_vals, lat_vals)
		xi = np.zeros((10201, 2))
		xi[:, 0] = xx.flatten()
		xi[:, 1] = yy.flatten()

		#Find depths
		bedrock_indices = findMatrixCoordsBedrock(florida_lon, florida_lat, xi[:,0], xi[:,1])
		bedrock_indices = np.array(bedrock_indices)
		depth_interpolated = florida_depth[bedrock_indices[:, 0], bedrock_indices[:, 1]]

		min_lon = np.min(xi[:,0])
		max_lon = np.max(xi[:,0])
		min_lat = np.min(xi[:,1])
		max_lat = np.max(xi[:,1])

		aot_869_interpolated = np.zeros((10201, len(sample_data_inds)))
		angstrom_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_412_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_443_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_469_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_488_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_531_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_547_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_555_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_645_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_667_interpolated = np.zeros((10201, len(sample_data_inds)))
		Rrs_678_interpolated = np.zeros((10201, len(sample_data_inds)))
		chlor_a_interpolated = np.zeros((10201, len(sample_data_inds)))
		chl_ocx_interpolated = np.zeros((10201, len(sample_data_inds)))
		Kd_490_interpolated = np.zeros((10201, len(sample_data_inds)))
		pic_interpolated = np.zeros((10201, len(sample_data_inds)))
		poc_interpolated = np.zeros((10201, len(sample_data_inds)))
		ipar_interpolated = np.zeros((10201, len(sample_data_inds)))
		nflh_interpolated = np.zeros((10201, len(sample_data_inds)))
		par_interpolated = np.zeros((10201, len(sample_data_inds)))
		for j in range(len(sample_data_inds)):
			file_id = data_list[sample_data_inds[j]]
			file_path = data_folder + '/' + file_id
			geo_dataset = xr.open_dataset(file_path, 'geophysical_data')
			nav_dataset = xr.open_dataset(file_path, 'navigation_data')
			latitude = nav_dataset['latitude']
			longitude = nav_dataset['longitude']
			latarr = np.array(latitude).flatten()
			longarr = np.array(longitude).flatten()

			lon_inds = (longarr > (min_lon-0.1)) & (longarr < (max_lon+0.1))
			lat_inds = (latarr > (min_lat-0.1)) & (latarr < (max_lat+0.1))
			valid_inds = np.logical_and(lon_inds, lat_inds)
			longarr = longarr[valid_inds]
			latarr = latarr[valid_inds]

			if(len(longarr) > 3):
				datapoints = np.zeros((len(longarr), 2))
				datapoints[:, 0] = longarr
				datapoints[:, 1] = latarr

				aot_869 = np.array(geo_dataset['aot_869']).flatten()
				aot_869 = aot_869[valid_inds]
				angstrom = np.array(geo_dataset['angstrom']).flatten()
				angstrom = angstrom[valid_inds]
				Rrs_412 = np.array(geo_dataset['Rrs_412']).flatten()
				Rrs_412 = Rrs_412[valid_inds]
				Rrs_443 = np.array(geo_dataset['Rrs_443']).flatten()
				Rrs_443 = Rrs_443[valid_inds]
				Rrs_469 = np.array(geo_dataset['Rrs_469']).flatten()
				Rrs_469 = Rrs_469[valid_inds]
				Rrs_488 = np.array(geo_dataset['Rrs_488']).flatten()
				Rrs_488 = Rrs_488[valid_inds]
				Rrs_531 = np.array(geo_dataset['Rrs_531']).flatten()
				Rrs_531 = Rrs_531[valid_inds]
				Rrs_547 = np.array(geo_dataset['Rrs_547']).flatten()
				Rrs_547 = Rrs_547[valid_inds]
				Rrs_555 = np.array(geo_dataset['Rrs_555']).flatten()
				Rrs_555 = Rrs_555[valid_inds]
				Rrs_645 = np.array(geo_dataset['Rrs_645']).flatten()
				Rrs_645 = Rrs_645[valid_inds]
				Rrs_667 = np.array(geo_dataset['Rrs_667']).flatten()
				Rrs_667 = Rrs_667[valid_inds]
				Rrs_678 = np.array(geo_dataset['Rrs_678']).flatten()
				Rrs_678 = Rrs_678[valid_inds]
				chlor_a = np.array(geo_dataset['chlor_a']).flatten()
				chlor_a = chlor_a[valid_inds]
				chl_ocx = np.array(geo_dataset['chl_ocx']).flatten()
				chl_ocx = chl_ocx[valid_inds]
				Kd_490 = np.array(geo_dataset['Kd_490']).flatten()
				Kd_490 = Kd_490[valid_inds]
				pic = np.array(geo_dataset['pic']).flatten()
				pic = pic[valid_inds]
				poc = np.array(geo_dataset['poc']).flatten()
				poc = poc[valid_inds]
				ipar = np.array(geo_dataset['ipar']).flatten()
				ipar = ipar[valid_inds]
				nflh = np.array(geo_dataset['nflh']).flatten()
				nflh = nflh[valid_inds]
				par = np.array(geo_dataset['par']).flatten()
				par = par[valid_inds]

				aot_869 = checkMODISBounds(aot_869, 'aot_869')
				angstrom = checkMODISBounds(angstrom, 'angstrom')
				Rrs_412 = checkMODISBounds(Rrs_412, 'Rrs_412')
				Rrs_443 = checkMODISBounds(Rrs_443, 'Rrs_443')
				Rrs_469 = checkMODISBounds(Rrs_469, 'Rrs_469')
				Rrs_488 = checkMODISBounds(Rrs_488, 'Rrs_488')
				Rrs_531 = checkMODISBounds(Rrs_531, 'Rrs_531')
				Rrs_547 = checkMODISBounds(Rrs_547, 'Rrs_547')
				Rrs_555 = checkMODISBounds(Rrs_555, 'Rrs_555')
				Rrs_645 = checkMODISBounds(Rrs_645, 'Rrs_645')
				Rrs_667 = checkMODISBounds(Rrs_667, 'Rrs_667')
				Rrs_678 = checkMODISBounds(Rrs_678, 'Rrs_678')
				chlor_a = checkMODISBounds(chlor_a, 'chlor_a')
				chl_ocx = checkMODISBounds(chl_ocx, 'chl_ocx')
				Kd_490 = checkMODISBounds(Kd_490, 'Kd_490')
				pic = checkMODISBounds(pic, 'pic')
				poc = checkMODISBounds(poc, 'poc')
				ipar = checkMODISBounds(ipar, 'ipar')
				nflh = checkMODISBounds(nflh, 'nflh')
				par = checkMODISBounds(par, 'par')

				vtx, wts = interp_weights(datapoints, xi)
				#interpolated_data = interpolate.griddata(datapoints, par, xi)
				interpolated_data = interpolate(aot_869, vtx, wts)
				aot_869_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(angstrom, vtx, wts)
				angstrom_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_412, vtx, wts)
				Rrs_412_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_443, vtx, wts)
				Rrs_443_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_469, vtx, wts)
				Rrs_469_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_488, vtx, wts)
				Rrs_488_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_531, vtx, wts)
				Rrs_531_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_547, vtx, wts)
				Rrs_547_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_555, vtx, wts)
				Rrs_555_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_645, vtx, wts)
				Rrs_645_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_667, vtx, wts)
				Rrs_667_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Rrs_678, vtx, wts)
				Rrs_678_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(chlor_a, vtx, wts)
				chlor_a_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(chl_ocx, vtx, wts)
				chl_ocx_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(Kd_490, vtx, wts)
				Kd_490_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(pic, vtx, wts)
				pic_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(poc, vtx, wts)
				poc_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(ipar, vtx, wts)
				ipar_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(nflh, vtx, wts)
				nflh_interpolated[:, j] = interpolated_data
				interpolated_data = interpolate(par, vtx, wts)
				par_interpolated[:, j] = interpolated_data
			else:
				aot_869_interpolated[:, j] = np.nan
				angstrom_interpolated[:, j] = np.nan
				Rrs_412_interpolated[:, j] = np.nan
				Rrs_443_interpolated[:, j] = np.nan
				Rrs_469_interpolated[:, j] = np.nan
				Rrs_488_interpolated[:, j] = np.nan
				Rrs_531_interpolated[:, j] = np.nan
				Rrs_547_interpolated[:, j] = np.nan
				Rrs_555_interpolated[:, j] = np.nan
				Rrs_645_interpolated[:, j] = np.nan
				Rrs_667_interpolated[:, j] = np.nan
				Rrs_678_interpolated[:, j] = np.nan
				chlor_a_interpolated[:, j] = np.nan
				chl_ocx_interpolated[:, j] = np.nan
				Kd_490_interpolated[:, j] = np.nan
				pic_interpolated[:, j] = np.nan
				poc_interpolated[:, j] = np.nan
				ipar_interpolated[:, j] = np.nan
				nflh_interpolated[:, j] = np.nan
				par_interpolated[:, j] = np.nan
		
		aot_869_mean = np.nanmean(aot_869_interpolated, axis=1)
		angstrom_mean = np.nanmean(angstrom_interpolated, axis=1)
		Rrs_412_mean = np.nanmean(Rrs_412_interpolated, axis=1)
		Rrs_443_mean = np.nanmean(Rrs_443_interpolated, axis=1)
		Rrs_469_mean = np.nanmean(Rrs_469_interpolated, axis=1)
		Rrs_488_mean = np.nanmean(Rrs_488_interpolated, axis=1)
		Rrs_531_mean = np.nanmean(Rrs_531_interpolated, axis=1)
		Rrs_547_mean = np.nanmean(Rrs_547_interpolated, axis=1)
		Rrs_555_mean = np.nanmean(Rrs_555_interpolated, axis=1)
		Rrs_645_mean = np.nanmean(Rrs_645_interpolated, axis=1)
		Rrs_667_mean = np.nanmean(Rrs_667_interpolated, axis=1)
		Rrs_678_mean = np.nanmean(Rrs_678_interpolated, axis=1)
		chlor_a_mean = np.nanmean(chlor_a_interpolated, axis=1)
		chl_ocx_mean = np.nanmean(chl_ocx_interpolated, axis=1)
		Kd_490_mean = np.nanmean(Kd_490_interpolated, axis=1)
		pic_mean = np.nanmean(pic_interpolated, axis=1)
		poc_mean = np.nanmean(poc_interpolated, axis=1)
		ipar_mean = np.nanmean(ipar_interpolated, axis=1)
		nflh_mean = np.nanmean(nflh_interpolated, axis=1)
		par_mean = np.nanmean(par_interpolated, axis=1)
		
		imagery = np.zeros((10201, 23))
		imagery[:, 0] = aot_869_mean
		imagery[:, 1] = angstrom_mean
		imagery[:, 2] = Rrs_412_mean
		imagery[:, 3] = Rrs_443_mean
		imagery[:, 4] = Rrs_469_mean
		imagery[:, 5] = Rrs_488_mean
		imagery[:, 6] = Rrs_531_mean
		imagery[:, 7] = Rrs_547_mean
		imagery[:, 8] = Rrs_555_mean
		imagery[:, 9] = Rrs_645_mean
		imagery[:, 10] = Rrs_667_mean
		imagery[:, 11] = Rrs_678_mean
		imagery[:, 12] = chlor_a_mean
		imagery[:, 13] = chl_ocx_mean
		imagery[:, 14] = Kd_490_mean
		imagery[:, 15] = pic_mean
		imagery[:, 16] = poc_mean
		imagery[:, 17] = ipar_mean
		imagery[:, 18] = nflh_mean
		imagery[:, 19] = par_mean
		imagery[:, 20] = xi[:,0]
		imagery[:, 21] = xi[:,1]
		imagery[:, 22] = depth_interpolated

		os.makedirs(save_folder+'{}'.format(i), exist_ok=True)
		np.save(save_folder+'{}/imagery.npy'.format(i), imagery)
		np.save(save_folder+'{}/conc.npy'.format(i), df_concs[i])
		np.save(save_folder+'{}/date.npy'.format(i), df_dates[i])