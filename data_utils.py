import pickle
import random
from sklearn.utils import shuffle as skshuffle
import numpy as np
from extract_data import normalize_data

class DataUtil:

    def __init__(self, data_dir, batch_size, num_epochs, normalize = True, supervised=True):


        if not supervised:
            self.images_train = list(pickle.load(open(data_dir + '/train_data.p', 'rb')))
            random.shuffle(self.images_train)

        else:
            self.images_train = list(pickle.load(open(data_dir + '/train_data.p', 'rb')))
            self.labels_train = list(pickle.load(open(data_dir + '/train_labels.p', 'rb')))

            images_val = list(pickle.load(open(data_dir + '/val_data.p', 'rb')))

            if normalize:
                images_val = normalize_data(images_val)

            self.images_val = images_val
            self.labels_val = list(pickle.load(open(data_dir + '/val_labels.p', 'rb')))



        self.batch_size = batch_size
        self.curr_data_num = 0
        self.global_num = 0

        self.curr_epoch = 0
        self.num_epochs = num_epochs

        self.supervised = supervised


    # def normalize_data(my_data, return_mean_and_std = False):
    #
    #     '''
    #     Args:
    #         2D array with arr storing each image, and arr[i] storing pixels of image i
    #     Returns:
    #         normalized my_data, mean of my_data, standard deviation of my_data
    #     '''
    #     m = np.mean(my_data, axis = 0)
    #     std = np.std(my_data, axis = 0)
    #
    #     my_data -= m
    #     my_data /= (std + 1e-8)
    #
    #     if return_mean_and_std:
    #         return my_data, m, std
    #     else:
    #         return my_data


    def get_next_batch(self):
        '''
        Returns:
            Next training batch, None if finished all epochs
        '''


        if self.supervised:
            if self.curr_epoch >= self.num_epochs:
                return None, None
            else:
                return self.get_next_batch_with_labels()

        else:
            if self.curr_epoch >= self.num_epochs:
                return None
            else:
                return self.get_next_batch_without_labels()



    def get_next_batch_without_labels(self):

        '''
        Gets the next batch in training data. WITHOUT LABELS
        @param None
        @return The next normalized training DATA BATCH
        '''

        img_batch = []

        for _ in range(self.batch_size):

            img_batch.append(self.images_train[self.curr_data_num])

            self.curr_data_num += 1
            self.global_num += 1

            if self.curr_data_num > len(self.images_train) - 1:

                print('FINISHED EPOCH', self.curr_epoch + 1)
                self.curr_epoch += 1
                self.curr_data_num = 0
                random.shuffle(self.images_train)


        img_batch = normalize_data(img_batch)

        return img_batch



    def get_next_batch_with_labels(self):
        '''
        Gets the next batch in training data. WITH LABELS
        @param None
        @return The next normalized training batch
        '''

        img_batch = []
        labels_batch = []

        for _ in range(self.batch_size):

            img_batch.append(self.images_train[self.curr_data_num])
            labels_batch.append(self.labels_train[self.curr_data_num])

            self.curr_data_num += 1
            self.global_num += 1

            if self.curr_data_num > len(self.images_train) - 1:

                print('FINISHED EPOCH', self.curr_epoch + 1)
                self.curr_epoch += 1
                self.curr_data_num = 0
                self.images_train, self.labels_train = skshuffle(self.images_train, self.labels_train)


        img_batch = normalize_data(img_batch)

        return img_batch, labels_batch




