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
import pandas as pd
import glob
import random


num_to_class = {
	0: 'None',
	1: 'thumb',
	2: 'index',
	3: 'middle',
	4: 'ring',
	5: 'pinkie'
}

class_to_num = {
    'None': 0,
    'thumb': 1,
    'index': 2,
    'middle': 3,
    'ring': 4,
    'pinkie': 5
}



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


    none = glob.glob('raw_data/none/emg*.csv')
    thumb = glob.glob('raw_data/thumb/emg*.csv')
    index = glob.glob('raw_data/index/emg*.csv')
    middle = glob.glob('raw_data/middle/emg*.csv')
    ring = glob.glob('raw_data/ring/emg*.csv')
    pinkie = glob.glob('raw_data/pinkie/emg*.csv')

    all_files = [none, thumb, index, middle, ring, pinkie]

    ALL_OF_THE_DATA = []
    ALL_OF_THE_LABELS = []

    for class_index, file_paths in enumerate(all_files):

        curr_class = num_to_class[class_index]
        all_data_per_class = []

        for path in file_paths:

            read_data = pd.read_csv(path)

            all_data = read_data.values.tolist()
            emg_data = [d[1:] for d in all_data]
            timestamps = [d[0] for d in all_data]



            #ignore the first and last two seconds of data
            emg_data = emg_data[400 : len(emg_data) - 400]
            timestamps = timestamps[400 : len(timestamps) - 400]


            all_image_data = []

            curr_image = []

            start_time = timestamps[0]
            data_index = 0
            while data_index < len(emg_data):

                #half a second
                if timestamps[data_index] - start_time >= 500000:

                    curr_image = reformat_curr_image(curr_image)
                    all_image_data.append(curr_image)
                    curr_image = []
                    start_time = timestamps[data_index]

                else:
                    curr_image += emg_data[data_index]


                data_index += 1

            #while loop done
            all_data_per_class += all_image_data

        #end of for loop for all files in that class dir
        ALL_OF_THE_DATA += all_data_per_class
        ALL_OF_THE_LABELS += (np.full(len(all_data_per_class), class_index)).tolist()


    print(len(ALL_OF_THE_DATA))
    print(len(ALL_OF_THE_LABELS))


    normalized, mean, std = normalize_data(ALL_OF_THE_DATA, True)

    with open('data/pop_mean.txt', 'w') as f:
        for m in mean:
            f.write(str(m) + '\n')

    with open('data/pop_std.txt', 'w') as f:
        for s in std:
            f.write(str(s) + '\n')


    pickle.dump(normalized, open('data/all_data.p', 'wb'))
    pickle.dump(ALL_OF_THE_LABELS, open('data/all_labels.p', 'wb'))



def random_scaling(data):

    for data_index, curr in enumerate(data):

        scale = random.uniform(0.8, 1.2)

        for val_index, val in enumerate(curr):
            data[data_index][val_index] = val * scale

    return data
        



def augment_data():

    original_data = pickle.load(open('data/all_data.p', 'rb')).tolist()
    all_labels = pickle.load(open('data/all_labels.p', 'rb'))


    augmented_1 = random_scaling(original_data)
    augmented_2 = random_scaling(augmented_1)
    augmented_3 = random_scaling(augmented_2)


    augmented_data = original_data + augmented_1 + augmented_2 + augmented_3
    all_labels = all_labels + all_labels + all_labels + all_labels

    pickle.dump(augmented_data, open('data/all_data.p', 'wb'))
    pickle.dump(all_labels, open('data/all_labels.p', 'wb'))


    


make_data()
augment_data()