import numpy as np
import matplotlib.pyplot as plt

data_folder = 'map_images2/'

dates = np.load(data_folder+'dates.npy', allow_pickle=True)

day_counter = 0
for day in dates:
	data = np.load(data_folder+'red_tide_output'+str(day_counter)+'.npy')

	land_mask = np.where(data[:,:,0]==-1)
	
	predicted_classes = np.argmax(data, axis=2)
	predicted_classes[land_mask[0], land_mask[1]] = -1

	plt.figure(dpi=500)
	ax = plt.gca()
	cax = ax.imshow(predicted_classes[220:440, 50:470].T)
	cbar = plt.colorbar(cax, ticks=[-1, 0, 1, 2, 3, 4, 5])
	cbar.ax.set_yticklabels(['Land', '0-1,000 cells/L', '1,000-10,000 cells/L', '10,000-50,000 cells/L', '50,000-100,000 cells/L', '100,000-1,000,000 cells/L', '1,000,000+ cells/L'])
	plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
	plt.title('{}/{}/{}'.format(day.month, day.day, day.year))
	plt.gca().invert_yaxis()
	plt.savefig(data_folder+'red_tide_image{}.png'.format(str(day_counter).zfill(5)), bbox_inches='tight')

	plt.close('all')

	day_counter += 1