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
	pickleFile = argv + '.pickle'
	with open(pickleFile, 'rb') as f:
		net = pickle.load(f)

	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])

	plt.plot(train_loss, linewidth=3, label="train")
	plt.plot(valid_loss, linewidth=3, label="valid")
	plt.grid()
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.ylim(1e-3, 1e-2)
	plt.yscale("log")
	figName = argv + 'TrainValidLoss.png'
	plt.savefig(figName)

if __name__ == "__main__":
	if len(argv) == 2:
		main(argv[1])
	else:
		print "Usage: python plotTrainValidError.py netname"