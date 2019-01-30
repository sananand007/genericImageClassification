'''
Using the CIFAR datasets
 - The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
 - 1024*3 = 3072
 - 10000 images in total
'''
import pickle
import os
from glob import glob
class procCifar(object):
	"""docstring for procCifar"""
	def __init__(self, path):
		super(procCifar, self).__init__()
		self.path = path

	def _unpickle(self):
		fnames, xtrain, testdir,xtest=[],[],[],[]
		for f in os.listdir(self.path):
			if 'data' in f:
				temp = self.path + f
				fnames.append(temp)
			if 'test' in f:
				testdir=self.path + f
		# Training set
		for i, f in enumerate(fnames):
			with open(f, 'rb') as f0:
				xtrain.append(pickle.load(f0, encoding='bytes'))
		
		# Test set
		with open(testdir, 'rb') as f0:
				xtest.append(pickle.load(f0, encoding='bytes'))


		print("shape of training dataset [1 batch]",xtrain[0][b'data'].shape)
		print("shape of testing dataset",xtest[0][b'data'].shape)
				

