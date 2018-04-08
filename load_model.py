import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import PIL.ImageOps
from PIL import Image
import glob
from matplotlib import pyplot as plt
from random import *
from data_utils import DataUtil

data_placeholder = tf.placeholder(shape = [None, 100*8], dtype = tf.float32)
label_placeholder = tf.placeholder(shape=[None], dtype = tf.int64)

labels = {
    0: 'none',
    1: 'thumb',
    2: 'index',
    3: 'middle',
    4: 'ring',
    5: 'pinkie'
}

labels_1 = {
    'none': 0,
    'thumb': 1,
    'index': 2,
    'middle': 3,
    'ring': 4,
    'pinkie': 5
}


MODEL_NAME = 'myo_piano_model'

def model(net):
    net = tf.reshape(net, [-1, 100, 8, 1])


    with tf.variable_scope(MODEL_NAME):
        with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
            with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
                
                net = slim.conv2d(net, 100, [5,5], scope='conv1')
                net = slim.max_pool2d(net, [2,2], scope='pool1')
                net = slim.batch_norm(net)
                net = slim.conv2d(net, 50, [5,5], scope='conv2')
                net = slim.max_pool2d(net, [2,2], scope='pool2')
                net = slim.batch_norm(net)
                net = slim.conv2d(net, 20, [5,5], scope='conv3')
                net = slim.max_pool2d(net, [2,2], scope='pool3')
                net = slim.batch_norm(net)
                net = slim.flatten(net, scope='flatten4')
                net = slim.fully_connected(net, 500, activation_fn = tf.nn.sigmoid, scope='fc5')
                net = slim.fully_connected(net, 6, activation_fn=None, scope='fc6')

    return net




def make_prediction(class_name):
    data = pickle.load(open('test_data/' + class_name + '.p', 'rb'))
    prediction = model(data_placeholder)

    print(len(data) , 'is length of dATA!!!')

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, 'out/myo_piano_model.chkp')

        logits = sess.run(prediction, feed_dict={data_placeholder: data})
        logits = tf.squeeze(logits)
        logits_arr = sess.run(logits)

        # print('LOGITS =', logits_arr)

        softmax_output = tf.nn.softmax(logits = logits_arr)
        probs = sess.run(softmax_output)
        # print('probs = ', probs)

        n = [np.argmax(x) for x in probs] 
        print(n)

        # try:
        #     print([labels[p] for p in n])
        # except:
        #     print(labels[n])

        sum=0
        nones=0
        for label in n:
            if label == labels_1[class_name]:
                sum += 1
            elif label == 0:
                nones += 1
            
        
        print('percent correct = ', sum/len(n))
        print('perc none=', nones/len(n))


make_prediction('pinkie')