import tensorflow as tf
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from get_csv_data import HandleData
from time import time

def corrupt(x, seed):
	r = tf.add(x, tf.cast(tf.random_uniform(shape=tf.shape(x),minval=0,maxval=0.1,dtype=tf.float32, seed=seed), tf.float32))
	return r

def autoencoder(dimensions=[784, 512, 256, 64], seed=1234):

	x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

	corrupt_prob = tf.placeholder(tf.float32, [1])
	current_input = corrupt(x, seed) * corrupt_prob + x * (1 - corrupt_prob)  # artificially corrupting the input signal
	noise_input = current_input
	# Build the encoder
	print("========= encoder begin ==========")
	encoder = []
	for layer_i, n_output in enumerate(dimensions[1:]):
		n_input = int(current_input.get_shape()[1])
		print("encoder : ", "n_layer",layer_i, "n_output",n_output, "n_input",n_input)
		W = tf.Variable(tf.random_uniform([n_input, n_output],-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input), seed=seed))
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)
		output = tf.nn.tanh(tf.matmul(current_input, W) + b)
		current_input = output
	print("========= encoder end =========")
	# latent representation
	z = current_input
	encoder.reverse()
	# Build the decoder using the same weights
	print("========= decoder begin ==========")
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
		print("decoder : ", "n_layer", layer_i,"n_output", n_output)
		W = tf.transpose(encoder[layer_i]) #  transpose of the weights
		b = tf.Variable(tf.zeros([n_output]))
		output = tf.nn.tanh(tf.matmul(current_input, W) + b)
		current_input = output
	print("========= decoder end =========")
	# now have the reconstruction through the network
	y = current_input
	# cost function measures pixel-wise difference
	cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
	return {
				'x': x,
				'z': z,
				'y': y,
				'corrupt_prob': corrupt_prob,
				'cost': cost,
				'noise_input' : noise_input
		   }

def getDAE(antenna_data=[], seed=1234):

	################ AutoEncoder ##############
	ae = autoencoder(dimensions=[4, 200], seed=seed)
	###########################################

	################ Training #################
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	########### restore ###########
	saver_restore = tf.train.import_meta_graph('./DAE_save/DenoisingAE_save_noise_add.meta')
	saver_restore.restore(sess, tf.train.latest_checkpoint('./DAE_save/'))
	###############################

	################ Testing trained data #####
	return_list = []
	for data in antenna_data:
		antenna_data_mean = np.mean(data, axis=0)
		test_xs_norm = np.array([img - antenna_data_mean for img in data])
		a,b,output_y = sess.run([ae['cost'],ae['noise_input'],ae['y']], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [1.0]})
		print("DEA avarage cost : ", a)
		return_list.append(output_y)
	tf.reset_default_graph()
	return return_list
	###########################################

def multilayer_perceptron(x, weights, biases):

	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'],name="DNN1")
	layer_1 = tf.nn.relu(layer_1,name="DNN2")
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'],name="DNN3")
	layer_2 = tf.nn.relu(layer_2,name="DNN4")
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['out'],name="DNN5") + biases['out']
	return out_layer

if __name__ == "__main__":
	test_percentage = 0.2

	seed = 1234

	data = HandleData(oneHotFlag=True)
	antenna_data, label_data = data.get_synthatic_data()
	antenna_data, antenna_data_test, label_data, label_test = train_test_split (antenna_data, label_data, test_size=test_percentage, random_state=42)

	DAE_out = getDAE([antenna_data, antenna_data_test], seed)  # get denoising autoencoder outputs for the train and test data

	#data.data_set = DAE_out[0]
	antenna_data = DAE_out[0]

	antenna_data_test = DAE_out[1]
	#data_test.data_set = DAE_out[1]

	TRAIN=True

	# Parameters
	learning_rate = 0.0001
	training_epochs = 2000
	batch_size = 5
	display_step = 1
	# Network Parameters
	n_hidden_1 = 12 # 1st layer number of features
	n_hidden_2 = 12 # 2nd layer number of features
	n_input = 4 # antenna_1, antenna_2, antenna_3, antenna_4
	n_classes = len(os.listdir('./Dround_Data_New/Nomalized')) # 0, 45, 90, 135, 180, 225, 270, 315

	# tf Graph input
	x = tf.placeholder("float", [None, n_input],name='DNN_x')
	y = tf.placeholder("float", [None, n_classes],name='DNN_y')

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=seed),name='DNN_w1'),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed=seed),name='DNN_w2'),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], seed=seed),name='DNN_w3')
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=seed),name='DNN_b1'),
		'b2': tf.Variable(tf.random_normal([n_hidden_2], seed=seed),name='DNN_b2'),
		'out': tf.Variable(tf.random_normal([n_classes], seed=seed),name='DNN_b3')
	}

	# Construct model
	pred = multilayer_perceptron(x, weights, biases)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name="DNN_cost")
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='DNN_optimizer').minimize(cost)

	# Initializing the Graph
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		if TRAIN:
			tic = time()
			############### Training #################
			for epoch in range(training_epochs):
				avg_cost = 0.
				total_batch = int(antenna_data.shape[0]/batch_size)
				for i in range(total_batch):
					batch_x = antenna_data[i*batch_size:i*batch_size + batch_size, :]
					batch_y = label_data[i*batch_size:i*batch_size + batch_size, :]
					_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
					avg_cost += c / total_batch
				# Display logs per epoch step
				if epoch % display_step == 0:
					print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
			print("Optimization Finished!")
			print('Elapsed time:', time() - tic, 'seconds')
			##########################################

			########## save ###########
			saver.save(sess, './DAEandDNN_save/DAEandDNN_save')
			###########################
		else:
			########### restore ###########
			saver_restore = tf.train.import_meta_graph('./DAEandDNN_save/DAEandDNN_save.meta')
			saver_restore.restore(sess, tf.train.latest_checkpoint('./DAEandDNN_save/'))
			###############################

		#### Calculate accuracy ###
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print("Accuracy (train data):", accuracy.eval({x: antenna_data, y: label_data}))

		print("Accuracy (test data):", accuracy.eval({x: antenna_data_test, y: label_test}))
		
		#label_test_pred = np.zeros((label_test.shape[0], label_test.shape[1]))
		label_test_pred = []
		label_test_list = []
		for i in range(antenna_data_test.shape[0]):
			x_i = antenna_data_test[i, :].reshape((1, antenna_data_test.shape[1]))
			y_i = label_test[i, :].reshape((1, label_test.shape[1]))
			pred_result = sess.run(tf.argmax(pred, 1), feed_dict={x: x_i, y: y_i})
			#one_hot = data.onehot_encode(pred_result[0])
			#label_test_pred[i, :] = one_hot
			label_test_pred.append(pred_result[0])
		for i in range(label_test.shape[0]):
			for j in range(label_test.shape[1]):
				if label_test[i, j] == 1: label_test_list.append(j)

		mat = confusion_matrix(label_test_list, label_test_pred)
		mat = np.round(mat / mat.astype(np.float).sum(axis=0) *100)
		mat = mat.astype(int)
		fig = plt.figure(figsize=(6, 6))
		sns.set()
		sns.heatmap(mat.T, square=True, annot=True, fmt='', cbar=False, xticklabels=['0\u00b0','45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0','270\u00b0','315\u00b0'], \
			yticklabels=['0\u00b0','45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0','270\u00b0','315\u00b0'], cmap="Blues")
		plt.xlabel('true label')
		plt.ylabel('predicted label')
		plt.savefig('confusion_matrix_dt_modified' + str(int(test_percentage*100)) + '%.png', dpi=600)
		plt.show()