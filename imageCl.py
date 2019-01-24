'''
Generic image Classification.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import argparse
import gc
import h5py
from collections import defaultdict
from tensorflow.python.framework import ops
from ops import norm


FLAGS=None
tf.logging.set_verbosity(tf.logging.INFO)


class genImageCl(object):
	def __init__(self, path, h5path):
		self.path = path
		# Print the contents of the path
		trainfiles, testfiles = [],[]
		to_scan = self.path+"*.csv"
		h5flag=False
		for f in glob(to_scan):
		    print(f)
		    if "train" in f: 
		        trainfiles.append(f)
		    elif "test" in f: 
		        testfiles.append(f)

		print(trainfiles, testfiles)
		# xtrain.iloc[0:0]
		# ytrain.iloc[0:0]
		# x_train.iloc[0:0]
		# y_train.iloc[0:0]
		to_scan_hd5 = to_scan = self.path+"*.h5"
		for fname in os.listdir(h5path):
			if fname.endswith('.h5'):
				h5flag=True
				break
		if not h5flag:
			xtrain,ytrain,self.x_train,self.y_train,xtest,ytest,self.x_test,self.y_test=[],[],[],[],[],[],[],[]
			xtrain = pd.read_csv(tf.gfile.Open(trainfiles[0]), nrows=150000, header=None) # change the nrows here
			ytrain = pd.read_csv(tf.gfile.Open(trainfiles[1]), nrows=150000, header=None) # change the nrows here
			self.x_train = xtrain.values.reshape((xtrain.shape[0],28,28,4)).clip(0,255).astype(np.float32)
			self.y_train = ytrain.values.astype(np.float32)
			xtest = pd.read_csv(tf.gfile.Open(testfiles[0]), nrows=60000, header=None)
			ytest = pd.read_csv(tf.gfile.Open(testfiles[1]), nrows=60000, header=None)
			self.x_test = xtest.values.reshape((xtest.shape[0],28,28,4)).clip(0,255).astype(np.float32)
			self.y_test = ytest.values.astype(np.float32)
			print(f'size of the sets, training set x {self.x_train.shape}, y {self.y_train.shape}')
			print(f'size of the sets, training set x {self.x_test.shape}, y {self.y_test.shape}')

			#Store the variables
			with h5py.File('x_train.h5','w') as hf:
				hf.create_dataset("train_x",  data=self.x_train)
			with h5py.File('y_train.h5','w') as hf:
				hf.create_dataset("train_y",  data=self.y_train)
			with h5py.File('x_test.h5','w') as hf:
				hf.create_dataset("test_x",  data=self.x_test)
			with h5py.File('y_test.h5','w') as hf:
				hf.create_dataset("test_y",  data=self.y_test)
		else:
			self.x_train,self.y_train,self.x_test,self.y_test=[],[],[],[]
			
			#Read them
			with h5py.File('x_train.h5','r') as hf:
				self.x_train = hf['train_x'][:]
			with h5py.File('y_train.h5','r') as hf:
				self.y_train = hf['train_y'][:]
			with h5py.File('x_test.h5','r') as hf:
				self.x_test = hf['test_x'][:]
			with h5py.File('y_test.h5','r') as hf:
				self.y_test = hf['test_y'][:]
			
			print(f'size of the sets, training set x {self.x_train.shape}, y {self.y_train.shape}')
			print(f'size of the sets, training set x {self.x_test.shape}, y {self.y_test.shape}')

	'''
	Without any Batch Normalization
	'''
	# Our application logic will be added here
	def cnn_model_fn(self, features, labels, mode, params, config):
	    #Input layer
	    input_layer = tf.reshape(features["x"], [-1, 28, 28, 4])
	    
	    # Convolutional Layer #1
	    conv1=tf.layers.conv2d(
	            inputs=input_layer,
	            filters=32,
	            kernel_size=[5,5],
	            padding="same",
	            activation=tf.nn.relu6)
	    
	    print("Shape Conv1:" + str(conv1.shape))
	    
	    # First Max Pooling layer
	    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2) #strides=2 . Divide size by 2
	    
	    print("Shape Pool1:" + str(pool1.shape))
	    
	    # Convolutional Layer #2
	    conv2=tf.layers.conv2d(
	            inputs=pool1,
	            filters=64,
	            kernel_size=[5,5],
	            padding="same",
	            activation=tf.nn.relu6)
	    
	    print("Shape Conv2:" + str(conv2.shape))
	    
	    # Second Max Pooling layer
	    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2) #strides=2 . Divide size by 2
	    
	    print("Shape Pool2:" + str(pool2.shape))
	    
	    #Flatten Pool 2
	    pool2_flat = tf.reshape(pool2, [-1, int(pool2.shape[1]) * int(pool2.shape[2]) * int(pool2.shape[3])])
	    
	    #Dense Layer
	    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu6)
	    
	    #Dropout
	    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	    
	    # Second Dense Layer
	    dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu6)
	    
	    #Output layer final
	    logits = tf.layers.dense(inputs=dense2, units=labels.shape[1])
	    
	    predictions = {
	        "classes": tf.argmax(input=logits, axis=1),
	        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
	        "logits":logits
	    }
	    
	    # Predict Mode
	    if mode==tf.estimator.ModeKeys.PREDICT:
	        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	    
	    # Loss Function
	    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
	    loss = tf.identity(loss, name="loss")
	    
	    
	    # Classification Metrics
	    # accuracy
	    acc  = tf.metrics.accuracy(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # Precision
	    prec = tf.metrics.precision(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # Recall
	    rec = tf.metrics.recall(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # F1 Score
	    f1 = 2 * acc[1] * rec[1] /(prec[1] + rec[1]) 
	    
	    
	    #TensorBoard Summary
	    with tf.name_scope('summaries'):
	        tf.summary.scalar('Accuracy', acc[1])
	        tf.summary.scalar('Precision', prec[1])
	        tf.summary.scalar('Recall', rec[1])
	        tf.summary.scalar('F1Score', f1)
	        tf.summary.histogram('Probabilities', predictions['probabilities'])
	        tf.summary.histogram('Classes', predictions['classes'])
	    
	    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(),save_steps=1)
	    
	    # Learning Rate Decay (Exponential)
	    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
	                                               global_step=tf.train.get_global_step(),
	                                               decay_steps=10000, 
	                                               decay_rate=0.96, 
	                                               staircase=True,
	                                               name='lr_exp_decay')
	    
	    # Configure the Training Op (for TRAIN mode)
	    if mode == tf.estimator.ModeKeys.TRAIN:
	        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	        train_op = optimizer.minimize(
	            loss=loss,
	            global_step=tf.train.get_global_step())
	        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	    
	    
	    # Evaluation Metrics
	    eval_metric_ops = {
	        "Accuracy": acc,
	        "Precision": prec,
	        "Recall": rec,
	    }
	    
	    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	def batch_norm_wrapper(self, inputs, is_training, decay = 0.999):
	    epsilon = 1e-3
	    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	    if is_training:
	        batch_mean, batch_var = tf.nn.moments(inputs,[0])
	        train_mean = tf.assign(pop_mean,
	                               pop_mean * decay + batch_mean * (1 - decay))
	        train_var = tf.assign(pop_var,
	                              pop_var * decay + batch_var * (1 - decay))
	        with tf.control_dependencies([train_mean, train_var]):
	            return tf.nn.batch_normalization(inputs,
	                batch_mean, batch_var, beta, scale, epsilon)
	    else:
	        return tf.nn.batch_normalization(inputs,
	            pop_mean, pop_var, beta, scale, epsilon)

	# Our application logic will be added here
	def cnn_model_bn_fn(self, features, labels, mode, params, config):
	    
	    #Input layer
	    input_layer = tf.reshape(features["x"], [-1, 28, 28, 4])
	    
	    # Convolutional Layer #1
	    conv1=tf.layers.conv2d(
	            inputs=input_layer,
	            filters=32,
	            kernel_size=[5,5],
	            padding="same",
	            activation=tf.nn.relu6)
	    
	    print("Shape Conv1:" + str(conv1.shape))
	    
	    # First Max Pooling layer
	    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2) #strides=2 . Divide size by 2
	    
	    print("Shape Pool1:" + str(pool1.shape))
	    
	    # Convolutional Layer #2
	    conv2=tf.layers.conv2d(
	            inputs=pool1,
	            filters=64,
	            kernel_size=[5,5],
	            padding="same",
	            activation=tf.nn.relu6)
	    
	    print("Shape Conv2:" + str(conv2.shape))
	    
	    # Second Max Pooling layer
	    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2) #strides=2 . Divide size by 2
	    
	    print("Shape Pool2:" + str(pool2.shape))
	    
	    #Flatten Pool 2
	    pool2_flat = tf.reshape(pool2, [-1, int(pool2.shape[1]) * int(pool2.shape[2]) * int(pool2.shape[3])])
	    
	    #Dense Layer
	    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu6)
	    
	    #Dropout
	    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	    
	    # Second Dense Layer
	    #dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu6)
	    
	    #Add Batch Normalization layer here
		#     if mode==tf.estimator.ModeKeys.TRAIN:
		#         batch_mean2, batch_var2 = tf.nn.moments(dropout,[0])
		#         scale2 = tf.Variable(tf.ones([1024]))
		#         beta2 = tf.Variable(tf.zeros([1024]))
		#         dense2 = tf.nn.batch_normalization(dropout,batch_mean2,batch_var2,beta2,scale2,epsilon)
		#     else:
		#         dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu6)
	    
	    # Second Dense Layer
	    dense2 = self.batch_norm_wrapper(dropout, is_training=mode == tf.estimator.ModeKeys.TRAIN)
	    
	    #Output layer final
	    logits = tf.layers.dense(inputs=dense2, units=labels.shape[1])
	    
	    predictions = {
	        "classes": tf.argmax(input=logits, axis=1),
	        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
	        "logits":logits
	    }
	    
	    # Predict Mode
	    if mode==tf.estimator.ModeKeys.PREDICT:
	        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	    
	    # Loss Function
	    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
	    loss = tf.identity(loss, name="loss")
	    
	    
	    # Classification Metrics
	    # accuracy
	    acc  = tf.metrics.accuracy(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # Precision
	    prec = tf.metrics.precision(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # Recall
	    rec = tf.metrics.recall(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # F1 Score
	    f1 = 2 * acc[1] * rec[1] /(prec[1] + rec[1]) 
	    
	    
	    #TensorBoard Summary
	    with tf.name_scope('summaries'):
	        tf.summary.scalar('Accuracy', acc[1])
	        tf.summary.scalar('Precision', prec[1])
	        tf.summary.scalar('Recall', rec[1])
	        tf.summary.scalar('F1Score', f1)
	        tf.summary.histogram('Probabilities', predictions['probabilities'])
	        tf.summary.histogram('Classes', predictions['classes'])
	    
	    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(),save_steps=1)
	    
	    # Learning Rate Decay (Exponential)
	    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
	                                               global_step=tf.train.get_global_step(),
	                                               decay_steps=10000, 
	                                               decay_rate=0.96, 
	                                               staircase=True,
	                                               name='lr_exp_decay')
	    
	    # Configure the Training Op (for TRAIN mode)
	    if mode == tf.estimator.ModeKeys.TRAIN:
	        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	        train_op = optimizer.minimize(
	            loss=loss,
	            global_step=tf.train.get_global_step())
	        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	    
	    
	    # Evaluation Metrics
	    eval_metric_ops = {
	        "Accuracy": acc,
	        "Precision": prec,
	        "Recall": rec,
	    }
	    
	    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	
	'''
	With Group Normalization + Fully connected
	'''

	def group_norm_wrapper(x, G=32, eps=1e-5, scope='group_norm') :
	    with tf.variable_scope(scope) :
	        N, H, W, C = x.get_shape().as_list()
	        print("Value inside the wrapper", N, H, W, C)
	        G = min(G, C)

	        x = tf.reshape(x, [N, H, W, G, C // G])
	        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
	        x = (x - mean) / tf.sqrt(var + eps)

	        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
	        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


	        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

	    return x

	# Our application logic will be added here
	def cnn_model_gn_fn(self,features, labels, mode, params, config):
	    
	    #Input layer
	    input_layer = tf.reshape(features["x"], [-1, 28, 28, 4])
	    
	    # Convolutional Layer #1
	    conv1=tf.layers.conv2d(
	            inputs=input_layer,
	            filters=32,
	            kernel_size=[5,5],
	            padding="same",
	            activation=tf.nn.relu6)
	    
	    print("Shape Conv1:" + str(conv1.shape))
	    
	    # First Max Pooling layer
	    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2) #strides=2 . Divide size by 2
	    
	    print("Shape Pool1:" + str(pool1.shape))
	    
	    # Convolutional Layer #2
	    conv2=tf.layers.conv2d(
	            inputs=pool1,
	            filters=64,
	            kernel_size=[5,5],
	            padding="same",
	            activation=tf.nn.relu6)
	    
	    print("Shape Conv2:" + str(conv2.shape))
	    
	    # Second Max Pooling layer
	    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2) #strides=2 . Divide size by 2
	    
	    print("Shape Pool2:" + str(pool2.shape))
	    
	    #Apply Group Normalization
	    x=norm(pool2, norm_type='group',is_train=True)
	    print("Shape after GN:" + str(x.shape))
	    
	    #Flatten Pool 2
	    pool2_flat = tf.reshape(x, [-1, int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])])
	    
	    #Dense Layer
	    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu6)
	    
	    #Dropout
	    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	    
	    # Second Dense Layer
	    dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu6)
	    
	    #Add Batch Normalization layer here
		#     if mode==tf.estimator.ModeKeys.TRAIN:
		#         batch_mean2, batch_var2 = tf.nn.moments(dropout,[0])
		#         scale2 = tf.Variable(tf.ones([1024]))
		#         beta2 = tf.Variable(tf.zeros([1024]))
		#         dense2 = tf.nn.batch_normalization(dropout,batch_mean2,batch_var2,beta2,scale2,epsilon)
		#     else:
		#         dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu6)
	    
	    #Output layer final
	    logits = tf.layers.dense(inputs=dense2, units=labels.shape[1])
	    
	    predictions = {
	        "classes": tf.argmax(input=logits, axis=1),
	        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
	        "logits":logits
	    }
	    
	    # Predict Mode
	    if mode==tf.estimator.ModeKeys.PREDICT:
	        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	    
	    # Loss Function
	    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
	    loss = tf.identity(loss, name="loss")
	    
	    
	    # Classification Metrics
	    # accuracy
	    acc  = tf.metrics.accuracy(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # Precision
	    prec = tf.metrics.precision(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # Recall
	    rec = tf.metrics.recall(labels=tf.argmax(labels,1), predictions=predictions['classes'])
	    
	    # F1 Score
	    f1 = 2 * acc[1] * rec[1] /(prec[1] + rec[1]) 
	    
	    
	    #TensorBoard Summary
	    with tf.name_scope('summaries'):
	        tf.summary.scalar('Accuracy', acc[1])
	        tf.summary.scalar('Precision', prec[1])
	        tf.summary.scalar('Recall', rec[1])
	        tf.summary.scalar('F1Score', f1)
	        tf.summary.scalar('loss', loss)
	        tf.summary.histogram('Probabilities', predictions['probabilities'])
	        tf.summary.histogram('Classes', predictions['classes'])
	    
	    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(),save_steps=1)
	    
	    # Learning Rate Decay (Exponential)
	    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
	                                               global_step=tf.train.get_global_step(),
	                                               decay_steps=10000, 
	                                               decay_rate=0.96, 
	                                               staircase=True,
	                                               name='lr_exp_decay')
	    
	    # Configure the Training Op (for TRAIN mode)
	    if mode == tf.estimator.ModeKeys.TRAIN:
	        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	        train_op = optimizer.minimize(
	            loss=loss,
	            global_step=tf.train.get_global_step())
	        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	    
	    
	    # Evaluation Metrics
	    eval_metric_ops = {
	        "Accuracy": acc,
	        "Precision": prec,
	        "Recall": rec,
	    }
	    
	    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)    

	'''
	Printing out the losses per step
	- Number of steps kept at 1 
	- Number of epochs kept at 3 each for train and test
	- 20 iterations for 50% of the training data and 75% of the test data 
	- With Batch Normalization used
	- Storing and printing the Loss and the accuracy to be plotted, this can be plotted on tensorboard
	'''
	def _runModel(self):
		# Rerun with larger number of steps
		from collections import defaultdict
		store_dict=defaultdict(dict)
		config = tf.ConfigProto(log_device_placement=True)
		loss_, accuracy_=[],[]

		sat6_classifier = tf.estimator.Estimator(model_fn=self.cnn_model_gn_fn, model_dir="/home/sandeeppanku/Public/deleteme/deepsat-sat6/3",
		                                         config=tf.contrib.learn.RunConfig(session_config=config))
		tensors_to_log={"probabilities":"softmax_tensor", "loss":"loss"}
		loss_.append(tensors_to_log["loss"])
		logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
		# Training input function
		train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": self.x_train},
		                                                    y=self.y_train,
		                                                    batch_size=512,
		                                                    num_epochs=3,
		                                                    shuffle=True)
		# Evaluation input function
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": self.x_test},
		                                                   y=self.y_test,
		                                                   num_epochs=3,
		                                                   shuffle=False)
		from tqdm import tqdm
		for i in tqdm(range(20)):
		    print(f"This is the {i} iteration")
		    sat6_classifier.train(input_fn=train_input_fn, steps=630, hooks=[logging_hook])
		    eval_results=sat6_classifier.evaluate(input_fn=eval_input_fn)
		    print(f"Results for {i} iteration {eval_results}")
		    store_dict.update(eval_results)

	def _usingtfgpu(self):
		'''
		Check if you have setup GPU correctly
		https://www.tensorflow.org/guide/using_gpu
		'''
		a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
		b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
		c = tf.matmul(a, b)
		
		# Creates a session with log_device_placement set to True.
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		
		# Runs the op.
		print(sess.run(c))

if __name__=="__main__":
	print(f"Entering the format")
	path = '/home/sandeeppanku/Public/deleteme/deepsat-sat6/'
	h5path = '/home/sandeeppanku/Public/Code/genericImageClassification/'
	gc.collect()
	obj=genImageCl(path, h5path)
	#obj._usingtfgpu()
	obj._runModel()