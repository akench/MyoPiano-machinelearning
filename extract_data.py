'''
Notes

Myoband data points unit = microseconds.

1 second is approximately ~200 data points
each data point has 8 values

EMG data in   ['emg']['data']   (list)
timestamps in ['emg']['timestamps']  (list)

index corresponds between two


Images will be 100 x 8   (height x width)   time axis is vertical height

'''


import json
from pprint import pprint
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt


def normalize_data(data, return_m_and_std = False):

	'''
	Args:
		2D array with arr storing each image, and arr[i] storing pixels of image i
	Returns:
		normalized data, mean of data, standard deviation of data
	'''
	m = np.mean(data, axis = 0)
	std = np.std(data, axis = 0)

	data -= m
	data /= (std + 1e-8)

	if return_m_and_std:
		return data, m, std
	else:
		return data


def reformat_curr_image(curr_image):

    while len(curr_image) < 800:
        curr_image.append(0)

    while len(curr_image) > 800:
        del curr_image[-1]

    return curr_image
    





def make_data():
    with open('first.json') as f:    
        all_data = json.load(f)


    emg_data = all_data['emg']['data']
    timestamps = all_data['emg']['timestamps']
    timestamps = [int(t) for t in timestamps]

    del all_data


    all_image_data = []

    curr_image = []

    start_time = timestamps[0]
    data_index = 0
    while data_index < len(emg_data):

        if timestamps[data_index] - start_time >= 500000:

            curr_image = reformat_curr_image(curr_image)
            all_image_data.append(curr_image)
            curr_image = []
            start_time = timestamps[data_index]

        else:
            curr_image += emg_data[data_index]


        data_index += 1


    normalized, mean, std = normalize_data(all_image_data, True)

    # print(normalized[0])

    for x in normalized:
        to_show = np.resize(x, (100, 8))

        plt.gray()
        plt.imshow(to_show)
        plt.show()

    # im = Image.fromarray(to_show)
    # im.show()




make_data()



