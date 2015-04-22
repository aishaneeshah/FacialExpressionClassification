import os

# Data Handling
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sys import argv

# Neural Net Imports
import theano

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator

import cPickle as pickle

from nnHeader import *
class AdjustVariable(object):

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
	flip_indices = [
		(0, 2), (1, 3),
		(4, 8), (5, 9), (6, 10), (7, 11),
		(12, 16), (13, 17), (14, 18), (15, 19), 
		(22, 24), (23, 25),
		]

	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs/2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		if yb is not None:
			yb[indices, ::2] = yb[indices, ::2]*-1

			for a, b in self.flip_indices:
				yb[indices, a], yb[indices, b] = (yb[indices, b], yb[indices, a])

		return Xb, yb

def main(argv):
	plotTrainStyle = ['r--','g--','b--','c--','m--','y--','k--']
	plotValidStyle = ['r','g','b','c','m','y','k']
	j = 0
	for pkl in argv:
		pickleFile = pkl + '.pickle'
		TrainTitle = pkl + ' Traning Error'
		ValidationTitle = pkl + ' Validation Error'
		with open(pickleFile, 'rb') as f:
			net = pickle.load(f)
		train_loss = np.array([i["train_loss"] for i in net.train_history_])
		valid_loss = np.array([i["valid_loss"] for i in net.train_history_])

		plt.plot(train_loss, plotTrainStyle[j], linewidth=2, label=TrainTitle)
		plt.plot(valid_loss, plotValidStyle[j], linewidth=3, label=ValidationTitle)

		j = (j+1)%7
	plt.grid()
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.ylim(1e-3, 1e-2)
	plt.yscale("log")
	figName = 'TrainValidLoss.png'
	plt.autoscale(enable=True, axis=u'both', tight=None)
	plt.savefig(figName)

if __name__ == "__main__":
	if len(argv) >= 2:
		main(argv[1:])
	else:
		print "Usage: python plotTrainValidError.py netname"
