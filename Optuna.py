import os

import numpy as np
import tensorflow as tf
from loguru import logger

from Train import train
from Model import EEGNet


class OptunaTrainer:
	def __init__(self, checkpointPath, epochs, batchsize, logPath=None):
		self.checkpointPath = checkpointPath
		self.logpath = logPath

		self.epochs = epochs
		self.batchsize = batchsize


	def __call__(self, trial, dataset, crossVal=False):
		if isinstance(dataset, tuple):
			dataset = {
				"noname": dataset
			}

		info = "Trial #{} auc values:".format(trial.number)
		auc = []
		for key, value in dataset.items():
			shape = value[0].shape[-2:]
			model = self.buildModel(trial, shape)
			_auc = train(
				model=model,
				dataset=value,
				weigthsPath=self.checkpointPath,
				epochs=self.epochs,
				batchsize=self.batchsize,
				crossVal=crossVal
			)
			info += "\t{}: {:.2f}".format(key, auc)

			auc.append(_auc)

		mean = np.mean(auc).round(2)
		median = np.median(auc).round(2)

		info += "\t Mean value: {}\tMedian value {}".format(mean, median)

		logger.info(info)
		logger.info(trial.params)

		return mean


	@staticmethod
	def chooseOptimizer(trial):
		kwargs = {}
		optimizer_options = ["RMSprop", "Adam"]

		optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)

		if optimizer_selected == "RMSprop":
			kwargs["learning_rate"] = trial.suggest_loguniform("rmsprop_learning_rate", 1e-5, 1e-1)
			kwargs["decay"] = trial.suggest_discrete_uniform("rmsprop_decay", 0.85, 0.99, 0.01)
			kwargs["momentum"] = trial.suggest_loguniform("rmsprop_momentum", 1e-5, 1e-1)
		elif optimizer_selected == "Adam":
			kwargs["learning_rate"] = trial.suggest_loguniform("adam_learning_rate", 1e-5, 1e-1)

		optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)

		return optimizer


	def buildModel(self, trial, shape):
		samples = shape[-1]
		assert samples // 2 > 16

		temporalLength = int(trial.suggest_discrete_uniform("temporal_length", 16, samples // 2, 4))
		dropoutRate = trial.suggest_discrete_uniform("dropout_rate", 0.1, 0.5, 0.05)
		D = trial.suggest_int("depth_multiplier", 1, 4)
		poolKernel = int(trial.suggest_discrete_uniform("pool_kernel", 4, 16, 2))

		model = EEGNet(
			categoriesN=2,
			electrodes=shape[0],
			samples=shape[1],
			temporalLength=temporalLength,
			dropoutRate=dropoutRate,
			D=D,
			poolPad="same",
			poolKernel=poolKernel
		)

		optimizer = self.chooseOptimizer(trial)

		model.compile(
			loss="sparse_categorical_crossentropy",
			optimizer=optimizer,
			metrics=["accuracy"]
		)

		return model


def studyInfo(study, bestN=7, file=None):
	logger.info("Number of finished trials: {}", len(study.trials))

	logger.info("Best {} trials:".format(bestN))
	trials = sorted(study.trials, key=lambda elem: elem.value, reverse=True)[:bestN]

	for i, trial in enumerate(trials):
		logger.info("Trial {}", i)
		logger.info("\tValue: {:.2f}", trial.value)

		logger.info("\tParams: ")
		for key, value in trial.params.items():
			logger.info("\t\t{}: {}", key, value)

	if file is not None:
		os.makedirs(os.path.dirname(file), exist_ok=True)
		studyDF = study.trials_dataframe()

		studyDF.to_csv(file)
		logger.info("Study file has been written to {}", file)