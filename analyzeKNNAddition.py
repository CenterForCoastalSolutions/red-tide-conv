import numpy as np
import matplotlib.pyplot as plt

allTestData = np.array([])
allTrueClasses = np.array([])
correctedSamples = np.array([])

for i in range(20):
	testData = np.load('saved_model_outputs/testData_wknn{}.npy'.format(i))
	trueClasses = np.load('saved_model_outputs/trueClasses_wknn{}.npy'.format(i))

	predictedClasseswknn = np.load('saved_model_outputs/predictedTargets_wknn{}.npy'.format(i))
	predictedClasseswoknn = np.load('saved_model_outputs/predictedTargets_woknn{}.npy'.format(i))

	wknnCorrect = np.where(trueClasses == predictedClasseswknn)[0]
	woknnIncorrect = np.where(trueClasses != predictedClasseswoknn)[0]

	correctedInds = np.intersect1d(wknnCorrect, woknnIncorrect)

	if(i == 0):
		correctedSamples = testData[correctedInds, :]
		allTrueClasses = trueClasses
		allTestData = testData
	else:
		correctedSamples = np.concatenate((correctedSamples, testData[correctedInds, :]))
		allTrueClasses = np.concatenate((allTrueClasses, trueClasses))
		allTestData = np.concatenate((allTestData, testData))

	#print(np.intersect1d(wknnCorrect, woknnIncorrect))

negativeSamples = np.where(allTrueClasses == 0)[0]
positiveSamples = np.where(allTrueClasses == 1)[0]

plt.figure(dpi=500)
plt.hist(allTestData[negativeSamples,2], bins=50, color='b', alpha=0.5, density=True, range=(0,20), label='Non-Bloom Conditions')
plt.hist(allTestData[positiveSamples,2], bins=50, color='r', alpha=0.5, density=True, range=(0,20), label='Bloom Conditions')
plt.hist(correctedSamples[:,2], bins=50, color='g', alpha=0.5, density=True, range=(0,20), label='Corrected Samples')
plt.title('Chlorophyll Concentration')
plt.xlabel('Feature Value')
plt.ylabel('Normalized Histogram')
plt.legend()
plt.savefig('chlor_feature.png', bbox_inches='tight')

plt.figure(dpi=500)
plt.hist(allTestData[negativeSamples,6], bins=50, color='b', alpha=0.5, density=True, range=(0,1), label='Non-Bloom Conditions')
plt.hist(allTestData[positiveSamples,6], bins=50, color='r', alpha=0.5, density=True, range=(0,1), label='Bloom Conditions')
plt.hist(correctedSamples[:,6], bins=50, color='g', alpha=0.5, density=True, range=(0,1), label='Corrected Samples')
plt.title('Normalized Fluorescence Line Height')
plt.xlabel('Feature Value')
plt.ylabel('Normalized Histogram')
plt.legend()
plt.savefig('nflh_feature.png', bbox_inches='tight')