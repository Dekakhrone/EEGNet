import os

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence

import config
from Utils.Augmentations import clipAxis
from Utils.DataLoader import splitDataset, crossValGenerator
from Utils.Metrics import ROC


class DataSequence(Sequence):
	def __init__(self, dataset, batchSize, sampleRate, augmenter=None, augProb=0.5, oversample=True, clip=(None, None)):
		self.dataset = dataset

		self.data, self.labels = self.dataset

		self.sampleRate = sampleRate

		self.batchSize = batchSize

		self.augmenter = augmenter
		self.augProb = augProb

		self.oversample = oversample
		self.clip = clip


	def __len__(self):
		return math.ceil(len(self.data) / self.batchSize)


	def __getitem__(self, idx):
		data = self.data[idx * self.batchSize:(idx + 1) * self.batchSize]
		labels = self.labels[idx * self.batchSize:(idx + 1) * self.batchSize]

		data = clipAxis(data, self.clip, self.sampleRate, axis=3)

		return data, labels


	def on_epoch_end(self):
		augment = np.random.choice([True, False], p=[self.augProb, 1 - self.augProb])

		if self.augmenter is not None and augment:
			self.data, self.labels = self.augmenter(*self.dataset, oversample=self.oversample, shuffle=True)
		else:
			self.data, self.labels = self.dataset


def test(model, checkpointPath, dataset, **kwargs):
	model.load_weights(checkpointPath)

	data, labels = dataset
	data = clipAxis(data, borders=kwargs.get("clip", (None, None)), sampleRate=config.sampleRate, axis=-1)

	pred = model.predict(data)[:, 1]
	auc = ROC(labels, pred,
	          show=kwargs.get("show", False), wpath=kwargs.get("wpath", None), name=kwargs.get("name", None))

	return auc


def train(model, dataset, weigthsPath, epochs=100, batchsize=128, crossVal=False, verbose=0, **kwargs):
	dataset = splitDataset(*dataset, trainPart=0.8, valPart=0.0)

	trainSet = dataset["train"]
	testSet = dataset["test"]

	if crossVal:
		trainSet = crossValGenerator(*trainSet, trainPart=0.8, valPart=0.2)
	else:
		trainSet = splitDataset(*trainSet, trainPart=0.8, valPart=0.2)
		trainSet = [trainSet]

	auc = []
	for i, set_ in enumerate(trainSet):
		checkpointPath = _train(
			model=model,
			dataset=set_,
			weigthsPath=weigthsPath,
			epochs=epochs,
			batchsize=batchsize,
			verbose=verbose,
			**kwargs
		)

		auc.append(test(model, checkpointPath, testSet, **kwargs))

		tf.keras.backend.clear_session()

	return np.mean(auc)


def _train(model, dataset, weigthsPath, epochs=100, batchsize=128, verbose=0, **kwargs):
	os.makedirs(weigthsPath, exist_ok=True)

	trainset = dataset["train"]
	valset = dataset["val"]

	trainset = DataSequence(trainset, batchsize, config.sampleRate, **kwargs)
	valset = (
		clipAxis(valset[0], borders=kwargs.get("clip", (None, None)), sampleRate=config.sampleRate, axis=-1),
		valset[1]
	)

	checkpointPath = os.path.join(weigthsPath, "best.h5")
	checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=verbose, save_best_only=True)

	model.fit(trainset, epochs=epochs, verbose=verbose, validation_data=valset, callbacks=[checkpointer])

	return checkpointPath