import pandas as pd
import sys
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import time
import math
import copy
from utils import *
from tqdm import tqdm
import datetime as dt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt
from nasnet_mobile import *
from nasnet_wrapper_model import *
from pixelwise_model import *
from pullConvData import *
from getTrainTestData import *
from dataset import *
from configparser import ConfigParser

configfilename = 'pixelwise+knn+dn+binned'

config = ConfigParser()
config.read('configfiles/'+configfilename+'.ini')

abovebelow50000 = config.getint('main', 'abovebelow50000')
randomseeds = json.loads(config.get('main', 'randomseeds'))
use_nn_feature = config.getint('main', 'use_nn_feature')
depth_normalize = config.getint('main', 'depth_normalize')
pixelwise = config.getint('main', 'pixelwise')
test_lit_methods = config.getint('main', 'test_lit_methods')
use_binned_data = config.getint('main', 'use_binned_data')

#positive_data, positive_concs, positive_dates, negative_data, negative_concs, negative_dates = pullConvData(1755, 1114)
if(abovebelow50000 == 0):
	data_folder = '/home/UFAD/rfick/Work/Github_repos/red-tide-conv/red_tide_conv_selected_data_correct/'
else:
	data_folder = '/home/UFAD/rfick/Work/Github_repos/red-tide-conv/red_tide_conv_selected_data_correct_abovebelow50000/'

model_number = 0

save_folder = 'saved_model_info/'+configfilename

features_to_use=['par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh', 'depth']

ensure_folder(save_folder)

for randomseed in randomseeds:
	print('Training model #{}'.format(model_number))

	# Set up random seeds for reproducability
	torch.manual_seed(randomseed)
	np.random.seed(randomseed)

	trainData, trainKNN, trainTargets, trainDates, testData, testKNN, testTargets, testDates = getTrainTestData(data_folder, features_to_use, use_nn_feature, depth_normalize, randomseed, use_binned_data, pixelwise)

	# Select 20% of training data points for validation
	nonValidationInds = np.random.choice(range(trainData.shape[0]), int(0.8*trainData.shape[0]), replace=False)
	validationInds = list(set(range(trainData.shape[0]))-set(nonValidationInds))

	if(pixelwise == 0):
		nonValidationData = trainData[nonValidationInds, :, :, :]
		validationData = trainData[validationInds, :, :, :]
		nonValidationKNN = trainKNN[nonValidationInds]
		validationKNN = trainKNN[validationInds]
		nonValidationTargets = trainTargets[nonValidationInds]
		validationTargets = trainTargets[validationInds]
	else:
		nonValidationData = trainData[nonValidationInds, :]
		validationData = trainData[validationInds, :]
		nonValidationKNN = trainKNN[nonValidationInds]
		validationKNN = trainKNN[validationInds]
		nonValidationKNN = np.expand_dims(nonValidationKNN, axis=1)
		validationKNN = np.expand_dims(validationKNN, axis=1)
		nonValidationData = np.concatenate((nonValidationData, nonValidationKNN), axis=1)
		validationData = np.concatenate((validationData, validationKNN), axis=1)
		nonValidationTargets = trainTargets[nonValidationInds]
		validationTargets = trainTargets[validationInds]

	learning_rate = 0.0001
	mb_size = 20
	numEpochs = 10000
	loss = nn.MSELoss()

	nonValidationDataTensor = torch.Tensor(nonValidationData).float().cuda()
	nonValidationKNNTensor = torch.Tensor(nonValidationKNN).float().cuda()
	nonValidationTargetsTensor = torch.Tensor(nonValidationTargets).float().cuda()
	validationDataTensor = torch.Tensor(validationData).float().cuda()
	validationKNNTensor = torch.Tensor(validationKNN).float().cuda()
	validationTargetsTensor = torch.Tensor(validationTargets).float().cuda()

	if(pixelwise == 0):
		nonValidationDataset = RedTideDataset(nonValidationDataTensor, nonValidationKNNTensor, nonValidationTargetsTensor)
		nonValidationDataloader = DataLoader(nonValidationDataset, batch_size=mb_size, shuffle=True)

		validationDataset = RedTideDataset(validationDataTensor, validationKNNTensor, validationTargetsTensor)
		validationDataloader = DataLoader(validationDataset, batch_size=mb_size, shuffle=True)

		#model = NASNetAMobile(num_classes=2).cuda()
		model = nasnet_wrapper_model(num_classes=2, use_nn_feature=use_nn_feature).cuda()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		#bestValModel = NASNetAMobile(num_classes=2).cuda()
		bestValModel = nasnet_wrapper_model(num_classes=2, use_nn_feature=use_nn_feature).cuda()
	else:
		nonValidationDataset = RedTideDatasetPixelwise(nonValidationDataTensor, nonValidationTargetsTensor)
		nonValidationDataloader = DataLoader(nonValidationDataset, batch_size=mb_size, shuffle=True)

		validationDataset = RedTideDatasetPixelwise(validationDataTensor, validationTargetsTensor)
		validationDataloader = DataLoader(validationDataset, batch_size=mb_size, shuffle=True)

		model = Predictor(input_dim=nonValidationDataTensor.shape[1], num_classes=2).cuda()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		bestValModel = Predictor(input_dim=nonValidationDataTensor.shape[1], num_classes=2).cuda()

	bestValidationLoss = math.inf
	epochsWithoutImprovement = 0

	for i in range(numEpochs):
		if(pixelwise == 0):
			model.eval()
			validationEpochLoss = 0
			with torch.no_grad():
				for mini_batch_data, mini_batch_knn, mini_batch_labels in validationDataloader:
					if(mini_batch_data.shape[0] == 1):
						continue
					output = model(mini_batch_data, mini_batch_knn)
					miniBatchLoss = loss(output, mini_batch_labels)
					validationEpochLoss += miniBatchLoss.item()

			model.train()
			nonValidationEpochLoss = 0
			for mini_batch_data, mini_batch_knn, mini_batch_labels in nonValidationDataloader:
				if(mini_batch_data.shape[0] == 1):
					continue
				optimizer.zero_grad()
				output = model(mini_batch_data, mini_batch_knn)
				miniBatchLoss = loss(output, mini_batch_labels)
				miniBatchLoss.backward()
				nonValidationEpochLoss += miniBatchLoss.item()
				optimizer.step()
		else:
			model.eval()
			validationEpochLoss = 0
			with torch.no_grad():
				for mini_batch_data, mini_batch_labels in validationDataloader:
					if(mini_batch_data.shape[0] == 1):
						continue
					output = model(mini_batch_data)
					miniBatchLoss = loss(output, mini_batch_labels)
					validationEpochLoss += miniBatchLoss.item()

			model.train()
			nonValidationEpochLoss = 0
			for mini_batch_data, mini_batch_labels in nonValidationDataloader:
				if(mini_batch_data.shape[0] == 1):
					continue
				optimizer.zero_grad()
				output = model(mini_batch_data)
				miniBatchLoss = loss(output, mini_batch_labels)
				miniBatchLoss.backward()
				nonValidationEpochLoss += miniBatchLoss.item()
				optimizer.step()

		print('Epoch: {}, Non Validation Loss: {}, Validation Loss: {}'.format(i, nonValidationEpochLoss, validationEpochLoss))

		if(validationEpochLoss < bestValidationLoss):
			bestValidationLoss = validationEpochLoss
			epochsWithoutImprovement = 0
			bestValModel = copy.deepcopy(model)
		else:
			epochsWithoutImprovement += 1
			bestValModel = copy.deepcopy(model)

		if(epochsWithoutImprovement > 50):
			print('Epoch {}, stopping early due to lack of improvement'.format(i))
			break

	torch.save(bestValModel.state_dict(), save_folder+'/'+str(randomseed)+'model_correct.pt')
	#np.save(save_folder+'/'+str(randomseed)+'testData_correct.npy', testData)
	#np.save(save_folder+'/'+str(randomseed)+'testTargets_correct.npy', testTargets)

	model_number = model_number + 1