import os

import numpy as np
import mne
import tensorflow as tf

from Model import EEGNet

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def reproduce(datasetDir, checkpointDir):
	from mne import io
	from mne.datasets import sample
	from tensorflow.keras import utils as np_utils
	from tensorflow.keras.callbacks import ModelCheckpoint

	for path in [datasetDir, checkpointDir]:
		os.makedirs(path, exist_ok=True)

	data_path = sample.data_path(datasetDir)

	# Set parameters and read data
	raw_fname = os.path.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')
	event_fname = os.path.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')
	tmin, tmax = -0., 1
	event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

	# Setup for reading the raw data
	raw = io.Raw(raw_fname, preload=True, verbose=False)
	raw.filter(2, None, method='iir')  # replace baselining with high-pass
	events = mne.read_events(event_fname)

	raw.info['bads'] = ['MEG 2443']  # set bad channels
	picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
	                       exclude='bads')

	# Read epochs
	epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
	                    picks=picks, baseline=None, preload=True, verbose=False)
	labels = epochs.events[:, -1]

	# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
	X = epochs.get_data() * 1000  # format is in (trials, channels, samples)
	y = labels

	kernels, electrodes, samples = 1, 60, 151

	# take 50/25/25 percent of the data to train/validate/test
	X_train = X[0:144, ...]
	Y_train = y[0:144]
	X_validate = X[144:216, ...]
	Y_validate = y[144:216]
	X_test = X[216:, ...]
	Y_test = y[216:]

	############################# EEGNet portion ##################################

	# convert labels to one-hot encodings.
	Y_train = np_utils.to_categorical(Y_train - 1)
	Y_validate = np_utils.to_categorical(Y_validate - 1)
	Y_test = np_utils.to_categorical(Y_test - 1)

	# convert data to NCHW (trials, kernels, channels, samples) format. Data
	# contains 60 channels and 151 time-points. Set the number of kernels to 1.
	X_train = X_train.reshape(X_train.shape[0], kernels, electrodes, samples)
	X_validate = X_validate.reshape(X_validate.shape[0], kernels, electrodes, samples)
	X_test = X_test.reshape(X_test.shape[0], kernels, electrodes, samples)

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
	# model configurations may do better, but this is a good starting point)
	model = EEGNet(categoriesN=4, electrodes=electrodes, samples=samples, dropoutRate=0.5, temporalLength=32, F1=8, D=2, F2=16)

	# compile the model and set the optimizers
	model.compile(loss='categorical_crossentropy', optimizer='adam',
	              metrics=['accuracy'])

	# count number of parameters in the model
	# numParams = model.count_params()

	# set a valid path for your system to record model checkpoints

	checkpointPath = os.path.join(checkpointDir, "best_mne_weights.h5")
	checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=1, save_best_only=True)

	model.fit(X_train, Y_train, batch_size=16, epochs=500,
	          verbose=2, validation_data=(X_validate, Y_validate), callbacks=[checkpointer])

	# load optimal weights
	model.load_weights(checkpointPath)

	###############################################################################
	# make prediction on test set.
	###############################################################################

	probs = model.predict(X_test)
	preds = probs.argmax(axis=-1)
	acc = np.mean(preds == Y_test.argmax(axis=-1))
	print("Classification accuracy: %f " % (acc))


def main():
	reproduce(
		datasetDir=r"D:\data\Research\mne_data",
		checkpointDir="./Data"
	)


if __name__ == "__main__":
	main()