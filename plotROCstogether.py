import numpy as np
import math
import matplotlib.pyplot as plt
from cycler import cycler
from os import listdir
from os.path import isfile, join

filename_roc_curve_info = 'roc_curve_info'
save_folder = 'roc_curve_plots'

files_to_display = ['pixelwise+knn+dn', 'paper_MLP', 'lit_methods_Tomlinson', 'lit_methods_Soto', 'lit_methods_RBDKBBI',\
					'lit_methods_Cannizzaro2008', 'lit_methods_Stumpf', 'lit_methods_RBD', 'lit_methods_Lou', 'lit_methods_Cannizzaro2009',\
					'lit_methods_Shehhi']

display_names = ['Spatio-temporal KNN + MLP', 'Hill et al. (2020)', 'Tomlinson et al. (2009)', 'Soto et al. (2021)', 'Amin et al. (2009) RBD+KBBI',\
					'Cannizzaro et al. (2008)', 'Stumpf et al. (2003)', 'Amin et al. (2009) RBD', 'Lou and Hu (2014)', 'Cannizzaro et al. (2009)',\
					'Al Shehhi et al. (2013)']

plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

all_means = []
all_stds = []
all_fprs = []

for file in files_to_display:
	filename = filename_roc_curve_info+'/'+file+'.npy'
	fpr_and_tprs = np.load(filename)

	fpr = fpr_and_tprs[:, 0]
	tpr_means = np.zeros(fpr_and_tprs.shape[0])
	tpr_stds = np.zeros(fpr_and_tprs.shape[0])
	for i in range(fpr_and_tprs.shape[0]):
		tpr_means[i] = np.mean(fpr_and_tprs[i, 1:])
		tpr_stds[i] = np.std(fpr_and_tprs[i, 1:])
	# Insert values to make sure plots start at (0, 0)
	fpr = np.insert(fpr, 0, 0)
	tpr_means = np.insert(tpr_means, 0, 0)
	tpr_stds = np.insert(tpr_stds, 0, 0)
	# Insert values to make sure plots end at (1, 1)
	fpr = np.append(fpr, 1)
	tpr_means = np.append(tpr_means, 1)
	tpr_stds = np.append(tpr_stds, 1)

	all_means.append(tpr_means)
	all_stds.append(tpr_stds)
	all_fprs.append(fpr)

plt.figure(dpi=500)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k')
plt.xlabel('False Positve Rate')
plt.ylabel('True Positive Rate')
plt.title('Model Comparison')

for i in range(len(all_means)):
	tpr_means = all_means[i]
	tpr_stds = all_stds[i]
	fpr = all_fprs[i]

	# margin of error for 95% confidence interval
	# margin of error = z*(population standard deviation/sqrt(n))
	# for 95% CI, z=1.96
	tpr_moes = 1.96*(tpr_stds/(math.sqrt(21)))
	markertouse = ''
	if(i <= 5):
		markertouse = 'solid'
	else:
		markertouse = 'dashed'
	plt.plot(fpr, tpr_means, label=display_names[i], color=plotColors[i%len(plotColors)], zorder=10, linewidth=3, linestyle=markertouse)
	x_values = np.concatenate((fpr, np.flip(fpr)))
	y_values = np.concatenate((tpr_means+tpr_moes, np.flip(tpr_means-tpr_moes)))
	#ax.fill(x_values, y_values, facecolor='blue', edgecolor='blue', alpha=0.3, linewidth=0, zorder=0)
	#ax.set_xticks([])
	#ax.set_yticks([])
	#ax.xaxis.set_label_position('top')
	#ax.set_xlabel(display_names[i], fontsize='4')

plt.tight_layout()
plt.legend(loc='lower right', prop={'size': 9})
plt.savefig(save_folder+'/combined_ROC.png', bbox_inches='tight')