import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger

import config
from Train import train
from Model import EEGNet


class OptunaTrainer:
	def __init__(self, checkpointPath, epochs, batchsize, logPath=None):
		self.checkpointPath = checkpointPath
		self.logpath = logPath

		self.epochs = epochs
		self.batchsize = batchsize


	def __call__(self, trial, dataset, crossVal=False, **kwargs):
		if isinstance(dataset, tuple):
			dataset = {
				"noname": dataset
			}

		info = "Trial #{} metric values:\n".format(trial.number)
		metrics = []
		for key, value in dataset.items():
			if "augmenter" in kwargs:
				kwargs["augmenter"].setState(False)

			shape = list(value[0].shape[-2:])
			shape[1] = int(config.window[1] * config.sampleRate) - int(config.window[0] * config.sampleRate)

			model = self.buildModel(trial, shape)
			auc, precision = train(
				model=model,
				dataset=value,
				weightsPath=self.checkpointPath,
				epochs=self.epochs,
				batchsize=self.batchsize,
				crossVal=crossVal,
				**kwargs
			)
			info += "{}: auc {:.2f} pr {:.2f}\t".format(key, auc, precision)

			metrics.append((auc, precision))

		metrics = np.array(metrics)

		mean = np.mean(metrics, axis=0).round(2)
		median = np.median(metrics, axis=0).round(2)

		for i, metric in enumerate(["auc", "precision"]):
			info += "\nMetric - {}. Mean: {}\tMedian: {}".format(metric, mean[i], median[i])

		logger.info(info)
		logger.info(trial.params)

		return mean[0]


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


	@staticmethod
	def chooseLoss(trial):
		loss_functions = {
			"binary_crossentropy": tf,
			"sigmoid_focal_crossentropy": tfa
		}

		loss_selected = trial.suggest_categorical("loss", list(loss_functions.keys()))
		loss = getattr(loss_functions[loss_selected].losses, loss_selected)

		return loss


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
		loss = self.chooseLoss(trial)

		model.compile(
			loss=loss,
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