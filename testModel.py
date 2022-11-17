import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from dataset import *
from nasnet_mobile import *
from nasnet_wrapper_model import *
from pixelwise_model import *
from getTrainTestData import *
from detectors.AminEtAlRBDDetector import *
from detectors.AminEtAlRBDKBBIDetector import *
from detectors.Cannizzaro2008EtAlDetector import *
from detectors.Cannizzaro2009EtAlDetector import *
from detectors.LouEtAlDetector import *
from detectors.ShehhiEtAlDetector import *
from detectors.SotoEtAlDetector import *
from detectors.StumpfEtAlDetector import *
from detectors.Tomlinson2009EtAlDetector import *
from configparser import ConfigParser
from ROCutils import *

configfilename = 'pixelwise+knn+dn+binned'

filename_roc_curve_info = 'roc_curve_info'

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

accs = np.zeros(len(randomseeds))
RBD_accs = np.zeros(len(randomseeds))
RBDKBBI_accs = np.zeros(len(randomseeds))
Cannizzaro2008_accs = np.zeros(len(randomseeds))
Cannizzaro2009_accs = np.zeros(len(randomseeds))
Lou_accs = np.zeros(len(randomseeds))
Shehhi_accs = np.zeros(len(randomseeds))
Soto_accs = np.zeros(len(randomseeds))
Stumpf_accs = np.zeros(len(randomseeds))
Tomlinson_accs = np.zeros(len(randomseeds))

kappas = np.zeros(len(randomseeds))
RBD_kappas = np.zeros(len(randomseeds))
RBDKBBI_kappas = np.zeros(len(randomseeds))
Cannizzaro2008_kappas = np.zeros(len(randomseeds))
Cannizzaro2009_kappas = np.zeros(len(randomseeds))
Lou_kappas = np.zeros(len(randomseeds))
Shehhi_kappas = np.zeros(len(randomseeds))
Soto_kappas = np.zeros(len(randomseeds))
Stumpf_kappas = np.zeros(len(randomseeds))
Tomlinson_kappas = np.zeros(len(randomseeds))

f1s = np.zeros(len(randomseeds))
RBD_f1s = np.zeros(len(randomseeds))
RBDKBBI_f1s = np.zeros(len(randomseeds))
Cannizzaro2008_f1s = np.zeros(len(randomseeds))
Cannizzaro2009_f1s = np.zeros(len(randomseeds))
Lou_f1s = np.zeros(len(randomseeds))
Shehhi_f1s = np.zeros(len(randomseeds))
Soto_f1s = np.zeros(len(randomseeds))
Stumpf_f1s = np.zeros(len(randomseeds))
Tomlinson_f1s = np.zeros(len(randomseeds))

refFpr = []
tprs = []
refFprRBD = []
tprsRBD = []
refFprRBDKBBI = []
tprsRBDKBBI = []
refFprCannizzaro2008 = []
tprsCannizzaro2008 = []
refFprCannizzaro2009 = []
tprsCannizzaro2009 = []
refFprLou = []
tprsLou = []
refFprShehhi = []
tprsShehhi = []
refFprSoto = []
tprsSoto = []
refFprStumpf = []
tprsStumpf = []
refFprTomlinson = []
tprsTomlinson = []

for randomseed in randomseeds:
	print('Testing model #{}'.format(model_number))

	# Set up random seeds for reproducability
	torch.manual_seed(randomseed)
	np.random.seed(randomseed)

	if(test_lit_methods == 0):
		features_to_use=['par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh', 'depth']
		trainData, trainKNN, trainTargets, trainDates, testData, testKNN, testTargets, testDates = getTrainTestData(data_folder, features_to_use, use_nn_feature, depth_normalize, randomseed, use_binned_data, pixelwise, test_only=True)

		if(pixelwise == 0):
			#model = NASNetAMobile(num_classes=2).cuda()
			model = nasnet_wrapper_model(num_classes=2, use_nn_feature=use_nn_feature).cuda()
			model.load_state_dict(torch.load(save_folder+'/'+str(randomseed)+'model_correct.pt'))
			model.eval()

			testDataTensor = torch.Tensor(testData).float().cuda()
			testKNNTensor = torch.Tensor(testKNN).float().cuda()
			testTargetsTensor = torch.Tensor(testTargets).float().cuda()

			mb_size = 10

			testDataset = RedTideDataset(testDataTensor, testKNNTensor, testTargetsTensor)
			testDataloader = DataLoader(testDataset, batch_size=mb_size, shuffle=False)

			predictedTargets = np.zeros_like(testTargets)
			targetIndex = 0

			for mini_batch_data, mini_batch_knn, mini_batch_labels in testDataloader:
				output = model(mini_batch_data, mini_batch_knn)
				predictedTargets[targetIndex:targetIndex+mb_size, :] = output.detach().cpu().numpy()
				targetIndex += mb_size
		else:
			testKNN = np.expand_dims(testKNN, axis=1)
			testData = np.concatenate((testData, testKNN), axis=1)
			testDataTensor = torch.Tensor(testData).float().cuda()
			testTargetsTensor = torch.Tensor(testTargets).float().cuda()

			model = Predictor(input_dim=testDataTensor.shape[1], num_classes=2).cuda()
			model.load_state_dict(torch.load(save_folder+'/'+str(randomseed)+'model_correct.pt'))
			model.eval()

			mb_size = 10

			testDataset = RedTideDatasetPixelwise(testDataTensor, testTargetsTensor)
			testDataloader = DataLoader(testDataset, batch_size=mb_size, shuffle=False)

			predictedTargets = np.zeros_like(testTargets)
			targetIndex = 0

			for mini_batch_data, mini_batch_labels in testDataloader:
				output = model(mini_batch_data)
				predictedTargets[targetIndex:targetIndex+mb_size, :] = output.detach().cpu().numpy()
				targetIndex += mb_size
	else:
		#Code goes here
		features_to_use=['Rrs_667', 'Rrs_678', 'chl_ocx', 'Rrs_443', 'Rrs_555', 'Rrs_488', 'nflh', 'chlor_a', 'Rrs_531']
		trainData, trainKNN, trainTargets, trainDates, testData, testKNN, testTargets, testDates = getTrainTestData(data_folder, features_to_use, use_nn_feature, depth_normalize, randomseed, pixelwise, test_only=True)

		AminRBDFeatures = [0, 1]
		CannizzaroFeatures = [2, 3, 4]
		LouFeatures = [3, 5, 4]
		ShehhiFeatures = [6]
		SotoFeatures = [2, 6, 3, 4]
		StumpfFeatures = [7]
		TomlinsonFeatures = [3, 5, 8]
		RBD = AminEtAlRBDDetector(testData[:, AminRBDFeatures])
		RBDKBBI = AminEtAlRBDKBBIDetector(testData[:, AminRBDFeatures])
		Cannizzaro2008 = Cannizzaro2008EtAlDetector(testData[:, CannizzaroFeatures])
		Cannizzaro2009 = Cannizzaro2009EtAlDetector(testData[:, CannizzaroFeatures])
		Lou = LouEtAlDetector(testData[:, LouFeatures])
		Shehhi = ShehhiEtAlDetector(testData[:, ShehhiFeatures])
		Soto = SotoEtAlDetector(testData[:, SotoFeatures])
		testDates = np.expand_dims(testDates, axis=1)
		StumpfInput = np.concatenate((testDates, testData[:, StumpfFeatures]), axis=1)
		Stumpf = StumpfEtAlDetector(StumpfInput)
		Tomlinson = Tomlinson2009EtAlDetector(testData[:, TomlinsonFeatures])


	trueClasses = np.argmax(testTargets, axis=1)

	if(test_lit_methods == 0):
		predictedClasses = np.argmax(predictedTargets, axis=1)

		accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFpr, tprs = addNewModelResultsROC(predictedTargets[:, 1], trueClasses, refFpr, tprs)
	else:
		predictedClasses = RBD > 0

		RBD_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		RBD_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		RBD_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprRBD, tprsRBD = addNewModelResultsROC(RBD, trueClasses, refFprRBD, tprsRBD)

		predictedClasses = RBDKBBI

		RBDKBBI_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		RBDKBBI_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		RBDKBBI_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprRBDKBBI, tprsRBDKBBI = addNewModelResultsROC(RBDKBBI, trueClasses, refFprRBDKBBI, tprsRBDKBBI)

		predictedClasses = Cannizzaro2008

		Cannizzaro2008_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Cannizzaro2008_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Cannizzaro2008_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprCannizzaro2008, tprsCannizzaro2008 = addNewModelResultsROC(Cannizzaro2008, trueClasses, refFprCannizzaro2008, tprsCannizzaro2008)

		predictedClasses = Cannizzaro2009

		Cannizzaro2009_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Cannizzaro2009_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Cannizzaro2009_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprCannizzaro2009, tprsCannizzaro2009 = addNewModelResultsROC(Cannizzaro2009, trueClasses, refFprCannizzaro2009, tprsCannizzaro2009)

		predictedClasses = Lou > 0

		Lou_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Lou_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Lou_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprLou, tprsLou = addNewModelResultsROC(Lou, trueClasses, refFprLou, tprsLou)

		predictedClasses = Shehhi > 0

		Shehhi_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Shehhi_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Shehhi_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprShehhi, tprsShehhi = addNewModelResultsROC(Shehhi, trueClasses, refFprShehhi, tprsShehhi)

		predictedClasses = Soto

		Soto_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Soto_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Soto_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprSoto, tprsSoto = addNewModelResultsROC(Soto, trueClasses, refFprSoto, tprsSoto)

		predictedClasses = Stumpf > 0

		Stumpf_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Stumpf_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Stumpf_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprStumpf, tprsStumpf = addNewModelResultsROC(Stumpf, trueClasses, refFprStumpf, tprsStumpf)

		predictedClasses = Tomlinson > 0

		Tomlinson_accs[model_number] = accuracy_score(trueClasses, predictedClasses)
		Tomlinson_kappas[model_number] = cohen_kappa_score(trueClasses, predictedClasses)
		Tomlinson_f1s[model_number] = f1_score(trueClasses, predictedClasses)

		refFprTomlinson, tprsTomlinson = addNewModelResultsROC(Tomlinson, trueClasses, refFprTomlinson, tprsTomlinson)

	model_number = model_number + 1

if(test_lit_methods == 0):
	print('Mean Accuracy: {}'.format(np.mean(accs)))
	print('Std Accuracy: {}'.format(np.std(accs)))
	print('Mean F1 Score: {}'.format(np.mean(f1s)))
	print('Std F1 Score: {}'.format(np.std(f1s)))
	print('Mean Cohen\'s Kappa Score: {}'.format(np.mean(kappas)))
	print('Std Cohen\'s Kappa Score: {}'.format(np.std(kappas)))

	refFpr = np.expand_dims(refFpr, axis=1)
	fpr_and_tprs = np.concatenate((refFpr, tprs), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'.npy', fpr_and_tprs)
else:
	print('RBD Mean Accuracy: {}'.format(np.mean(RBD_accs)))
	print('RBD Std Accuracy: {}'.format(np.std(RBD_accs)))
	print('RBD Mean F1 Score: {}'.format(np.mean(RBD_f1s)))
	print('RBD Std F1 Score: {}'.format(np.std(RBD_f1s)))
	print('RBD Mean Cohen\'s Kappa Score: {}'.format(np.mean(RBD_kappas)))
	print('RBD Std Cohen\'s Kappa Score: {}'.format(np.std(RBD_kappas)))

	refFprRBD = np.expand_dims(refFprRBD, axis=1)
	fpr_and_tprsRBD = np.concatenate((refFprRBD, tprsRBD), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_RBD.npy', fpr_and_tprsRBD)

	print('RBDKBBI Mean Accuracy: {}'.format(np.mean(RBDKBBI_accs)))
	print('RBDKBBI Std Accuracy: {}'.format(np.std(RBDKBBI_accs)))
	print('RBDKBBI Mean F1 Score: {}'.format(np.mean(RBDKBBI_f1s)))
	print('RBDKBBI Std F1 Score: {}'.format(np.std(RBDKBBI_f1s)))
	print('RBDKBBI Mean Cohen\'s Kappa Score: {}'.format(np.mean(RBDKBBI_kappas)))
	print('RBDKBBI Std Cohen\'s Kappa Score: {}'.format(np.std(RBDKBBI_kappas)))

	refFprRBDKBBI = np.expand_dims(refFprRBDKBBI, axis=1)
	fpr_and_tprsRBDKBBI = np.concatenate((refFprRBDKBBI, tprsRBDKBBI), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_RBDKBBI.npy', fpr_and_tprsRBDKBBI)

	print('Cannizzaro2008 Mean Accuracy: {}'.format(np.mean(Cannizzaro2008_accs)))
	print('Cannizzaro2008 Std Accuracy: {}'.format(np.std(Cannizzaro2008_accs)))
	print('Cannizzaro2008 Mean F1 Score: {}'.format(np.mean(Cannizzaro2008_f1s)))
	print('Cannizzaro2008 Std F1 Score: {}'.format(np.std(Cannizzaro2008_f1s)))
	print('Cannizzaro2008 Mean Cohen\'s Kappa Score: {}'.format(np.mean(Cannizzaro2008_kappas)))
	print('Cannizzaro2008 Std Cohen\'s Kappa Score: {}'.format(np.std(Cannizzaro2008_kappas)))

	refFprCannizzaro2008 = np.expand_dims(refFprCannizzaro2008, axis=1)
	fpr_and_tprsCannizzaro2008 = np.concatenate((refFprCannizzaro2008, tprsCannizzaro2008), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Cannizzaro2008.npy', fpr_and_tprsCannizzaro2008)

	print('Cannizzaro2009 Mean Accuracy: {}'.format(np.mean(Cannizzaro2009_accs)))
	print('Cannizzaro2009 Std Accuracy: {}'.format(np.std(Cannizzaro2009_accs)))
	print('Cannizzaro2009 Mean F1 Score: {}'.format(np.mean(Cannizzaro2009_f1s)))
	print('Cannizzaro2009 Std F1 Score: {}'.format(np.std(Cannizzaro2009_f1s)))
	print('Cannizzaro2009 Mean Cohen\'s Kappa Score: {}'.format(np.mean(Cannizzaro2009_kappas)))
	print('Cannizzaro2009 Std Cohen\'s Kappa Score: {}'.format(np.std(Cannizzaro2009_kappas)))

	refFprCannizzaro2009 = np.expand_dims(refFprCannizzaro2009, axis=1)
	fpr_and_tprsCannizzaro2009 = np.concatenate((refFprCannizzaro2009, tprsCannizzaro2009), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Cannizzaro2009.npy', fpr_and_tprsCannizzaro2009)

	print('Lou Mean Accuracy: {}'.format(np.mean(Lou_accs)))
	print('Lou Std Accuracy: {}'.format(np.std(Lou_accs)))
	print('Lou Mean F1 Score: {}'.format(np.mean(Lou_f1s)))
	print('Lou Std F1 Score: {}'.format(np.std(Lou_f1s)))
	print('Lou Mean Cohen\'s Kappa Score: {}'.format(np.mean(Lou_kappas)))
	print('Lou Std Cohen\'s Kappa Score: {}'.format(np.std(Lou_kappas)))

	refFprLou = np.expand_dims(refFprLou, axis=1)
	fpr_and_tprsLou = np.concatenate((refFprLou, tprsLou), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Lou.npy', fpr_and_tprsLou)

	print('Shehhi Mean Accuracy: {}'.format(np.mean(Shehhi_accs)))
	print('Shehhi Std Accuracy: {}'.format(np.std(Shehhi_accs)))
	print('Shehhi Mean F1 Score: {}'.format(np.mean(Shehhi_f1s)))
	print('Shehhi Std F1 Score: {}'.format(np.std(Shehhi_f1s)))
	print('Shehhi Mean Cohen\'s Kappa Score: {}'.format(np.mean(Shehhi_kappas)))
	print('Shehhi Std Cohen\'s Kappa Score: {}'.format(np.std(Shehhi_kappas)))

	refFprShehhi = np.expand_dims(refFprShehhi, axis=1)
	fpr_and_tprsShehhi = np.concatenate((refFprShehhi, tprsShehhi), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Shehhi.npy', fpr_and_tprsShehhi)

	print('Soto Mean Accuracy: {}'.format(np.mean(Soto_accs)))
	print('Soto Std Accuracy: {}'.format(np.std(Soto_accs)))
	print('Soto Mean F1 Score: {}'.format(np.mean(Soto_f1s)))
	print('Soto Std F1 Score: {}'.format(np.std(Soto_f1s)))
	print('Soto Mean Cohen\'s Kappa Score: {}'.format(np.mean(Soto_kappas)))
	print('Soto Std Cohen\'s Kappa Score: {}'.format(np.std(Soto_kappas)))

	refFprSoto = np.expand_dims(refFprSoto, axis=1)
	fpr_and_tprsSoto = np.concatenate((refFprSoto, tprsSoto), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Soto.npy', fpr_and_tprsSoto)

	print('Stumpf Mean Accuracy: {}'.format(np.mean(Stumpf_accs)))
	print('Stumpf Std Accuracy: {}'.format(np.std(Stumpf_accs)))
	print('Stumpf Mean F1 Score: {}'.format(np.mean(Stumpf_f1s)))
	print('Stumpf Std F1 Score: {}'.format(np.std(Stumpf_f1s)))
	print('Stumpf Mean Cohen\'s Kappa Score: {}'.format(np.mean(Stumpf_kappas)))
	print('Stumpf Std Cohen\'s Kappa Score: {}'.format(np.std(Stumpf_kappas)))

	refFprStumpf = np.expand_dims(refFprStumpf, axis=1)
	fpr_and_tprsStumpf = np.concatenate((refFprStumpf, tprsStumpf), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Stumpf.npy', fpr_and_tprsStumpf)

	print('Tomlinson Mean Accuracy: {}'.format(np.mean(Tomlinson_accs)))
	print('Tomlinson Std Accuracy: {}'.format(np.std(Tomlinson_accs)))
	print('Tomlinson Mean F1 Score: {}'.format(np.mean(Tomlinson_f1s)))
	print('Tomlinson Std F1 Score: {}'.format(np.std(Tomlinson_f1s)))
	print('Tomlinson Mean Cohen\'s Kappa Score: {}'.format(np.mean(Tomlinson_kappas)))
	print('Tomlinson Std Cohen\'s Kappa Score: {}'.format(np.std(Tomlinson_kappas)))

	refFprTomlinson = np.expand_dims(refFprTomlinson, axis=1)
	fpr_and_tprsTomlinson = np.concatenate((refFprTomlinson, tprsTomlinson), axis=1)

	np.save(filename_roc_curve_info+'/'+configfilename+'_Tomlinson.npy', fpr_and_tprsTomlinson)