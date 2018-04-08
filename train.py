import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import numpy as np
import time
import pickle
from sklearn.utils import shuffle
import os
import os.path as path
from extract_data import normalize_data
from data_utils import DataUtil


MODEL_NAME = 'myo_piano_model'
data_placeholder = tf.placeholder(shape = [None, 100*8], dtype = tf.float32, name='input')
labels_placeholder = tf.placeholder(shape = [None], dtype = tf.int64)

logs_path = "/tmp/myo_log"
#command to use TENSORBOARD
#tensorboard --logdir=run1:/tmp/myo_log/ --port 6006

import os
import glob

files = glob.glob('/tmp/myo_log/test/*')
files += glob.glob('/tmp/myo_log/train/*')

for f in files:
	os.remove(f)



labels = {
	0: 'None',
	1: 'thumb',
	2: 'index',
	3: 'middle',
	4: 'ring',
	5: 'pinkie'
}




def model(net):
	net = tf.reshape(net, [-1, 100, 8, 1])

	tf.summary.image('input', net, 10)

	with tf.variable_scope(MODEL_NAME, reuse=tf.AUTO_REUSE):
		with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
				
				net = slim.conv2d(net, 100, [20,1], scope='conv1')
				net = slim.max_pool2d(net, [2,2], scope='pool1')
				net = slim.batch_norm(net)
				net = slim.conv2d(net, 50, [20,1], scope='conv2')
				net = slim.max_pool2d(net, [2,2], scope='pool2')
				net = slim.batch_norm(net)
				net = slim.conv2d(net, 20, [20,1], scope='conv3')
				net = slim.max_pool2d(net, [2,2], scope='pool3')
				net = slim.batch_norm(net)
				net = slim.flatten(net, scope='flatten4')
				net = slim.fully_connected(net, 500, activation_fn = tf.nn.sigmoid, scope='fc5')
				net = slim.fully_connected(net, 6, activation_fn=None, scope='fc6')

	output = tf.identity(net, name='output')
	return net




def train():

	data_util = DataUtil('processed_data', batch_size = 128, num_epochs = 20)



	prediction = model(data_placeholder)

	with tf.name_scope('total_loss'):
		total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits = prediction, labels = labels_placeholder))

	with tf.name_scope('train_step'):
		optimizer = tf.train.AdamOptimizer()
		train_step = optimizer.minimize(total_loss)

	with tf.name_scope('accuracy'):
		correct = tf.equal(tf.argmax(prediction, 1), labels_placeholder)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	tf.summary.scalar("total_loss", total_loss)
	tf.summary.scalar("accuracy", accuracy)
	# tf.summary.scalar("train_step", optimizer)




	#MERGES ALL SUMMARIES INTO ONE OPERATION
	#THIS CAN BE EXECUTED IN A SESSION
	summary_op = tf.summary.merge_all()

	with tf.Session() as sess:
		start_time = time.time()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		init.run()

		tf.train.write_graph(sess.graph_def, 'out', MODEL_NAME + '.pbtxt', True)

		img_batch, labels_batch = data_util.get_next_batch()
		while img_batch is not None:

			if data_util.global_num % 100 == 0:
				print('curr acc:',accuracy.eval({data_placeholder: data_util.images_val[:100],
											labels_placeholder: data_util.labels_val[:100]}))

			'''ACTUAL TRAINING'''
			

			_, summary = sess.run([train_step, summary_op],
												feed_dict = {data_placeholder: img_batch,
												labels_placeholder: labels_batch})



			img_batch, labels_batch = data_util.get_next_batch()








		#TRAINING DONE!!!!!!!!!!!!!!
		#VAL IMAGES ALREADY NORMALIZED
		print('\n\nfinal Accuracy:',accuracy.eval({data_placeholder: data_util.images_val,
											labels_placeholder: data_util.labels_val}))


		


		print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - start_time)))

		save_path = saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
		print("path saved in", save_path)




def main():
	if not path.exists('out'):
		os.mkdir('out')

	input_node_name = 'input'
	output_node_name = 'output'

	train()

if __name__ == '__main__':
	main()
