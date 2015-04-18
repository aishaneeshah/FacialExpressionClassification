import os

# Data Handling
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import theano

# Neural Net Imports
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import cPickle as pickle

FTRAIN = './data/training.csv'
FTEST = './data/test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2D(test = False, cols = None):
    X, y = load(test = test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
 
 
#	class FlipBatchIterator(BatchIterator):
#
#		flip_indices = [
#			(0, 2), (1, 3),
#			(4, 8), (5, 9), (6, 10), (7, 11),
#			(12, 16), (13, 17), (14, 18), (15, 19), 
#			(22, 24), (23, 25),
#			]
#
#		def transform(self, Xb, yb):
#			Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
#
#			bs = Xb.shape[0]
#			indices = np.random.choice(bs, bs/2, replace=False)
#			Xb[indices] = Xb[indices, :, :, ::-1]
#
#			if yb is not None:
#				yb[indices, ::2] = yb[indices, ::2]*-1
#
#				for a, b in self.flip_indices:
#					yb[indices, a], yb[indices, b] = (yb[indices, b], yb[indices, a])
#
#			return Xb, yb
#

def float32(k):
    return np.cast['float32'](k)
		
		
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

		
# net2 = NeuralNet(
#     layers=[
#         ('input', layers.InputLayer),
#         ('conv1', layers.Conv2DLayer),
#         ('pool1', layers.MaxPool2DLayer),
#         ('conv2', layers.Conv2DLayer),
#         ('pool2', layers.MaxPool2DLayer),
#         ('conv3', layers.Conv2DLayer),
#         ('pool3', layers.MaxPool2DLayer),
#         ('hidden4', layers.DenseLayer),
#         ('hidden5', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     input_shape=(None, 1, 96, 96),
#     conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
#     conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
#     conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
#     hidden4_num_units=500, hidden5_num_units=500,
#     output_num_units=30, output_nonlinearity=None,

#     update_learning_rate=0.01,
#     update_momentum=0.9,

#     regression=True,
#     max_epochs=1000,
#     verbose=1,
#     )

#	net3 = NeuralNet(
#		layers=[
#			('input', layers.InputLayer),
#			('conv1', layers.Conv2DLayer),
#			('pool1', layers.MaxPool2DLayer),
#			('conv2', layers.Conv2DLayer),
#			('pool2', layers.MaxPool2DLayer),
#			('conv3', layers.Conv2DLayer),
#			('pool3', layers.MaxPool2DLayer),
#			('hidden4', layers.DenseLayer),
#			('hidden5', layers.DenseLayer),
#			('output', layers.DenseLayer),
#			],
#		input_shape=(None, 1, 96, 96),
#		conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
#		conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
#		conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
#		hidden4_num_units=500, hidden5_num_units=500,
#		output_num_units=30, output_nonlinearity=None,
#
#		update_learning_rate=0.01,
#		update_momentum=0.9,
#
#		regression=True,
#		max_epochs=1000,
#		verbose=1,
#
#		batch_iterator_train = FlipBatchIterator(batch_size = 128),
#		)
#
	
net4 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

	update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
	on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=1000,
    verbose=1,
	
    )	
	
X, y = load2D()
net4.fit(X, y)
#net3.fit(X, y)
# net2.fit(X, y)

with open('net4.pickle', 'wb') as f:
    pickle.dump(net4, f, -1)

# with open('net3.pickle', 'wb') as f:
#    pickle.dump(net3, f, -1)

# with open('net2.pickle', 'rb') as f:
# 	net2 = pickle.load(f)

# sample2 = load2D(test=True)[0][6:7]
# y_pred2 = net2.predict(sample2)[0]

# fig = plt.figure(figsize=(6, 3))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
# plot_sample(sample2[0], y_pred2, ax)
# plt.savefig('test.png')
# plt.show()
