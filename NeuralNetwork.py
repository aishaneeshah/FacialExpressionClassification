from Include import *
import os
import sys


sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)

FTRAIN = './data/training.csv'
FTEST = './data/test.csv'
FLOOKUP = './data/IdLookupTable.csv'
SAVESTATE = './data/save_netstate.pickle'
SAVEWEIGHTS = './data/save_netweights.pickle'

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
	    ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
    dropout3_p=0.5,
    hidden4_num_units=1000,
    dropout4_p=0.5,
    hidden5_num_units=1000,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    max_epochs=1850,
    verbose=1,
    )

X, y = load2d()

try:
    f = open(SAVESTATE, 'rb')
    state = pickle.load(f)
    net.__setstate__(state)
    # net.max_epochs = 2000
    # net.on_epoch_finished=[
    #     AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #     AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #     EarlyStopping(patience=200),
    #     ]
    print "Continuing"

except:
    print "Failed Set State or Starting First Time"

net.fit(X, y)

state = net.__getstate__()
with open(SAVESTATE, 'wb') as f:
    pickle.dump(state, f, -1)