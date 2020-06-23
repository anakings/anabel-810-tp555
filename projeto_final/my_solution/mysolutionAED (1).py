#I used also the DecisionTree model but it gives a lower accuracy
#I used also their autoencoder and decoder but it gives a lower accuracy for my model (BaggingClassifier) 
import tensorflow as tf
import math
import numpy as np
from get_csv_data import HandleData
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns 
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

if __name__ == '__main__':
	test_percentage = 0.2

	seed = 1234

	#instance of the Handle Data class
	data = HandleData(oneHotFlag=False)
	#get the data
	antenna_data, label_data = data.get_synthatic_data()
	antenna_data, antenna_data_test, label_data, label_test = train_test_split (antenna_data, label_data, \
		test_size=test_percentage, random_state=42)

	DAE_out = getDAE([antenna_data, antenna_data_test], seed)  # get denoising autoencoder outputs for the train and test data
	antenna_data = DAE_out[0]
	antenna_data_test = DAE_out[1]

	#Instantiate a Bagging Classifier
	clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=300, max_samples=250, \
		bootstrap=False, n_jobs=-1, random_state=42)
	#Train model
	tic = time()
	clf.fit(antenna_data, label_data)
	#Predict
	y_pred = clf.predict(antenna_data_test)
			
	print('Accuracy of model is:', accuracy_score(label_test, y_pred)*100, '%')
	print('Elapsed time:', time() - tic, 'seconds')
	
	mat = confusion_matrix(label_test, y_pred)
	mat = np.round(mat / mat.astype(np.float).sum(axis=0) *100)
	mat = mat.astype(int)
	fig = plt.figure(figsize=(6, 6))
	sns.set()
	sns.heatmap(mat.T, square=True, annot=True, fmt='', cbar=False, xticklabels=['0\u00b0','45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0','270\u00b0','315\u00b0'], \
		yticklabels=['0\u00b0','45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0','270\u00b0','315\u00b0'], cmap="Blues")
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.savefig('confusion_matrix_dt_mysolutionAED' + str(int(test_percentage*100)) + '%.png', dpi=600)
	plt.show()