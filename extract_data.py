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


            # for offset in [10, 50, 90, 130]:
            for offset in [0]:
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

                        data_index += offset

                    else:
                        curr_image += emg_data[data_index]
                        data_index += 1

                #while loop done
                all_data_per_class += all_image_data

        #end of for loop for all files in that class dir
        ALL_OF_THE_DATA += all_data_per_class
        ALL_OF_THE_LABELS += (np.full(len(all_data_per_class), class_index)).tolist()



    _, mean, std = normalize_data(ALL_OF_THE_DATA, True)

    with open('data/pop_mean.txt', 'w') as f:
        for m in mean:
            f.write(str(m) + '\n')

    with open('data/pop_std.txt', 'w') as f:
        for s in std:
            f.write(str(s) + '\n')


    print(len(ALL_OF_THE_LABELS))
    ALL_OF_THE_DATA, ALL_OF_THE_LABELS = simultaneous_shuffle(ALL_OF_THE_DATA, ALL_OF_THE_LABELS)

    pickle.dump(ALL_OF_THE_DATA, open('data/all_data.p', 'wb'))
    pickle.dump(ALL_OF_THE_LABELS, open('data/all_labels.p', 'wb'))



def random_scaling(data):

    for data_index, curr in enumerate(data):

        scale = random.uniform(0.8, 1.2)

        for val_index, val in enumerate(curr):
            data[data_index][val_index] = val * scale

    return data
        



def augment_data():

    original_data = pickle.load(open('data/all_data.p', 'rb'))
    all_labels = pickle.load(open('data/all_labels.p', 'rb'))


    augmented_1 = random_scaling(original_data)

    augmented_data = original_data + augmented_1
    all_labels = all_labels + all_labels


    pickle.dump(original_data, open('data/all_data.p', 'wb'))
    pickle.dump(all_labels, open('data/all_labels.p', 'wb'))


    print('augmented',len(augmented_data))

    return




def normalize_using_pop_data(data):

    with open('data/pop_mean.txt', 'r') as f:
        mean_list = [float(m) for m in f.readlines()]
    with open('data/pop_std.txt', 'r') as f:
        std_list = [float(s) for s in f.readlines()]

    normalized_data = []
    
    for img in data:

        normalized_img = []
        for i, val in enumerate(img):

            normalized_img.append((val - mean_list[i]) / std_list[i])

        normalized_data.append(normalized_img)


    return normalized_data






def make_data_per_class_testing(class_name):

    file_paths = glob.glob('data/' + class_name + '/emg*.csv')

    ALL_IMAGES = []

    #list of csv files
    for path in file_paths:

        all_data = pd.read_csv(path).values.tolist()

        emg_data = [d[1:] for d in all_data]
        timestamps = [d[0] for d in all_data]



        #ignore the first and last one second of data
        emg_data = emg_data[200 : len(emg_data) - 200]
        timestamps = timestamps[200 : len(timestamps) - 200]

        for offset in [0]:

            curr_image = []

            start_time = timestamps[0]
            data_index = 0

            #rows in csv file
            while data_index < len(emg_data):

                #half a second
                if timestamps[data_index] - start_time >= 500000:

                    curr_image = reformat_curr_image(curr_image)
                    ALL_IMAGES.append(curr_image)
                    curr_image = []
                    start_time = timestamps[data_index]

                    data_index += offset

                else:
                    curr_image += emg_data[data_index]
                    data_index += 1

    
    ALL_IMAGES = normalize_using_pop_data(ALL_IMAGES)
    pickle.dump(ALL_IMAGES, open('test_data/' + class_name + '.p', 'wb'))




def split_data(save_folder, all_data, all_labels, perc_train = 0.80, perc_val = 0.1, perc_test = 0.1):
    num_data = len(all_data)
    num_train = int(perc_train * num_data)
    num_val = int(perc_val * num_data)


    curr = 0
    train_data = all_data[curr : num_train]
    train_labels = all_labels[curr : num_train]
    pickle.dump(train_data, open(save_folder + '/train_data.p', 'wb'))
    pickle.dump(train_labels, open(save_folder + '/train_labels.p', 'wb'))

    curr += num_train
    val_data = all_data[curr : curr + num_val]
    val_labels = all_labels[curr : curr + num_val]
    pickle.dump(val_data, open(save_folder + '/val_data.p', 'wb'))
    pickle.dump(val_labels, open(save_folder + '/val_labels.p', 'wb'))

    curr += num_val
    test_data = all_data[curr:]
    test_labels = all_labels[curr:]
    pickle.dump(test_data, open(save_folder + '/test_data.p', 'wb'))
    pickle.dump(test_labels, open(save_folder + '/test_labels.p', 'wb'))


def simultaneous_shuffle(A, B):

    C = list(zip(A, B))
    random.shuffle(C)

    A, B = zip(*C)

    return A,B




def prepare_data_to_split():

    data = pickle.load(open('data/all_data.p', 'rb'))
    labels = pickle.load(open('data/all_labels.p', 'rb'))

    data, labels = simultaneous_shuffle(data, labels)

    split_data('processed_data', data, labels)



# make_data()
# augment_data()
# prepare_data_to_split()

# for x in ['none', 'thumb', 'index', 'middle', 'ring', 'pinkie']:
#     make_data_per_class_testing(x)