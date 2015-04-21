import os

# Data Handling
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

# Neural Net Imports
import theano

from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator

import cPickle as pickle

FTRAIN = './data/training.csv'
FTEST = './data/test.csv'
FLOOKUP = './data/IdLookupTable.csv'
FSUBMIT = './data/Submission.csv'

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

SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]

def fit_specialists(fname_pretrain=None):
    if fname_pretrain:
        with open(fname_pretrain, 'rb') as f:
            net_pretrain = pickle.load(f)
    else:
        net_pretrain = None

    specialists = OrderedDict()

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)

        model = clone(net)
        model.output_num_units = y.shape[1]
        model.batch_iterator_train.flip_indices = setting['flip_indices']
        model.max_epochs = int(4e6 / y.shape[0])
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])

        if net_pretrain is not None:
            # if a pretrain model was given, use it to initialize the
            # weights of our new specialist model:
            model.load_weights_from(net_pretrain)

        print("Training model for columns {} for {} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model

    with open('net-specialists.pickle', 'wb') as f:
        # this time we're persisting a dictionary with all models:
        pickle.dump(specialists, f, -1)

def predict(fname_specialists='net-specialists.pickle'):
    with open(fname_specialists, 'rb') as f:
        specialists = pickle.load(f)

    X = load2d(test=True)[0]
    y_pred = np.empty((X.shape[0], 0))

    for model in specialists.values():
        y_pred1 = model.predict(X)
        y_pred = np.hstack([y_pred, y_pred1])

    columns = ()
    for cols in specialists.keys():
        columns += cols

    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns=columns)

    lookup_table = read_csv(os.path.expanduser(FLOOKUP))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))

def rebin( a, newshape ):
    from numpy import mgrid
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def plot_learning_curves(fname_specialists='net-specialists.pickle'):
    with open(fname_specialists, 'r') as f:
        models = pickle.load(f)

    fig = pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_color_cycle(
        ['c', 'c', 'm', 'm', 'y', 'y', 'k', 'k', 'g', 'g', 'b', 'b'])

    valid_losses = []
    train_losses = []

    for model_number, (cg, model) in enumerate(models.items(), 1):
        valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
        train_loss = np.array([i['train_loss'] for i in model.train_history_])
        valid_loss = np.sqrt(valid_loss) * 48
        train_loss = np.sqrt(train_loss) * 48

        valid_loss = rebin(valid_loss, (100,))
        train_loss = rebin(train_loss, (100,))

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)
        ax.plot(valid_loss,
                label='{} ({})'.format(cg[0], len(cg)), linewidth=3)
        ax.plot(train_loss,
                linestyle='--', linewidth=3, alpha=0.6)
        ax.set_xticks([])

    weights = np.array([m.output_num_units for m in models.values()],
                       dtype=float)
    weights /= weights.sum()
    mean_valid_loss = (
        np.vstack(valid_losses) * weights.reshape(-1, 1)).sum(axis=0)
    ax.plot(mean_valid_loss, color='r', label='mean', linewidth=4, alpha=0.8)

    ax.legend()
    ax.set_ylim((1.0, 4.0))
    ax.grid()
    pyplot.ylabel("RMSE")
    pyplot.show()

with open('net4.pickle', 'rb') as f:
	net4 = pickle.load(f)

X, _ = load2D(test=True)
# net4.fit(X, y)

y_pred = net4.predict(X)
y_pred = y_pred*48 + 48
y_pred = clip(y_pred, (0, 96))

LookUp = {  'left_eye_center_x':0, 'left_eye_center_y':1, 
            'right_eye_center_x':2, 'right_eye_center_y':3,
            'left_eye_inner_corner_x':4, 'left_eye_inner_corner_y':5,
            'left_eye_outer_corner_x':6, 'left_eye_outer_corner_y':7,
            'right_eye_inner_corner_x':8, 'right_eye_inner_corner_y':9,
            'right_eye_outer_corner_x':10, 'right_eye_outer_corner_y':11,
            'left_eyebrow_inner_end_x':12, 'left_eyebrow_inner_end_y':13,
            'left_eyebrow_outer_end_x':14, 'left_eyebrow_outer_end_y':15,
            'right_eyebrow_inner_end_x':16, 'right_eyebrow_inner_end_y':17,
            'right_eyebrow_outer_end_x':18, 'right_eyebrow_outer_end_y':19,
            'nose_tip_x':20, 'nose_tip_y':21,
            'mouth_left_corner_x':22, 'mouth_left_corner_y':23,
            'mouth_right_corner_x':24, 'mouth_right_corner_y':25,
            'mouth_center_top_lip_x':26, 'mouth_center_top_lip_y':27,
            'mouth_center_bottom_lip_x':28, 'mouth_center_bottom_lip_x':29
        }

with open(FLOOKUP, 'r') as csvinput:
    with open(FSUBMIT, 'w') as csvoutput:
        reader = csv.DictReader(csvinput)
        fieldnames = ['RowId', 'ImageId', 'FeatureName', 'Location']
        writer = csv.DictWriter(csvoutput, fieldnames=fieldnames)

        for row in reader:
            write.writerow({'RowId':row['RowId'], 'ImageId':row['ImageId'], 'FeatureName':row['FeatureName'],
                            'Location':y_pred[row['ImageId'] - 1, LookUp['FeatureName']]})