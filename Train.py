import os

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence

import config
from Utils.Augmentations import clipAxis, getOversampler, prob2bool
from Utils.DataLoader import splitDataset, crossValGenerator
from Utils.Metrics import ROC, average_precision


oversampler = getOversampler()


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
		augment = prob2bool(self.augProb)

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

	precision = average_precision(labels, pred)

	return auc, precision


def train(model, dataset, weightsPath, epochs=100, batchsize=128, crossVal=False, weightedLoss=True,
          verbose=0, **kwargs):
	dataset = splitDataset(*dataset, trainPart=0.8, valPart=0.0)

	trainSet = dataset["train"]
	testSet = oversampler.oversample(*dataset["test"]) # it's necessary to get balanced test set always

	if crossVal:
		trainSet = crossValGenerator(*trainSet, trainPart=0.8, valPart=0.2)
	else:
		trainSet = splitDataset(*trainSet, trainPart=0.8, valPart=0.2)
		trainSet = [trainSet]

	metrics = []
	for i, set_ in enumerate(trainSet):
		checkpointPath = _train(
			model=model,
			dataset=set_,
			weightsPath=weightsPath,
			epochs=epochs,
			batchsize=batchsize,
			verbose=verbose,
			weightedLoss=weightedLoss,
			**kwargs
		)

		metrics.append(test(model, checkpointPath, testSet, **kwargs))

		tf.keras.backend.clear_session()

	metrics = np.array(metrics)

	return np.mean(metrics, axis=0)


def _train(model, dataset, weightsPath, epochs=100, batchsize=128, weightedLoss=True, verbose=0, **kwargs):
	os.makedirs(weightsPath, exist_ok=True)

	trainset = dataset["train"]
	valset = oversampler.oversample(*dataset["val"]) # it's necessary to get balanced val set always
	labels = trainset[1]

	trainset = DataSequence(trainset, batchsize, config.sampleRate, **kwargs)
	valset = (
		clipAxis(valset[0], borders=kwargs.get("clip", (None, None)), sampleRate=config.sampleRate, axis=-1),
		valset[1]
	) # in case if augmentations was applied before

	checkpointPath = os.path.join(weightsPath, "best.h5")
	checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=verbose, save_best_only=True)

	_, counts = np.unique(labels, return_counts=True)
	if weightedLoss and model.loss == "binary_crossentropy":
		sampleWeights = [1 - count / np.sum(counts) for count in counts]
		sampleWeights = [coef / np.max(sampleWeights) for coef in sampleWeights]
	else:
		sampleWeights = [1 for _ in counts]

	sampleWeights = np.tile(sampleWeights, batchsize).reshape(batchsize, len(counts))

	model.fit(trainset, epochs=epochs, verbose=verbose, validation_data=valset, callbacks=[checkpointer],
	          loss_weights=sampleWeights)

	return checkpointPath