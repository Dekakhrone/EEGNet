import os
import yaml

import optuna
import numpy as np
import tensorflow as tf
from loguru import logger
from functools import partial
from tensorflow.keras.callbacks import ModelCheckpoint

from Model import EEGNet
from Utils.DataLoader import DataHandler, Formats, splitDataset, crossValGenerator
from Utils.Metrics import ROC


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


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
		aucSum = 0
		for key, value in dataset.items():
			auc = self.handle(trial, value, crossVal=crossVal)
			info += "\t{}: {:.2f}".format(key, auc)

			aucSum += auc

		logger.info(info)

		return aucSum / len(dataset)


	def handle(self, trial, dataset, crossVal=False):
		shape = dataset[0].shape[-2:]
		model = self.buildModel(trial, shape)
		# model.save_weights(os.path.join(self.checkpointPath, "temp_weights.hdf"))

		dataset = splitDataset(*dataset, trainPart=0.8, valPart=0.0)

		trainSet = dataset["train"]
		testSet = dataset["test"]

		dataset = crossValGenerator(*trainSet, trainPart=0.8, valPart=0.2) if crossVal else [dataset]

		auc = 0
		for i, set_ in enumerate(dataset):
			checkpointPath = train(
				model=model,
				dataset=set_,
				weigthsPath=self.checkpointPath,
				logPath=self.logpath,
				epochs=self.epochs,
				batchsize=self.batchsize
			)

			auc += test(model, checkpointPath, testSet)

			tf.keras.backend.clear_session()
			# model.load_weights(os.path.join(self.checkpointPath, "temp_weights.hdf"))

		return auc / (i + 1)


	@staticmethod
	def _loadConfig(configSource):
		if isinstance(configSource, dict):
			return configSource
		elif isinstance(configSource, str):
			pass
		else:
			raise TypeError

		with open(configSource, "r", encoding="utf-8") as stream:
			config = yaml.safe_load(stream)

		assert config is not None

		return config


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


def test(model, checkpointPath, dataset):
	model.load_weights(checkpointPath)

	data, labels = dataset

	pred = model.predict(data)[:, 1]
	auc = ROC(labels, pred)

	return auc


def train(model, dataset, weigthsPath, logPath=None, epochs=100, batchsize=128):
	for path in [weigthsPath, logPath]:
		if path is not None:
			os.makedirs(path, exist_ok=True)

	checkpointPath = os.path.join(weigthsPath, "best.h5")
	checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=0, save_best_only=True)

	model.fit(*dataset["train"], batch_size=batchsize, epochs=epochs, verbose=0, validation_data=dataset["val"],
	          callbacks=[checkpointer])

	return checkpointPath


def studyInfo(study):
	logger.info("Number of finished trials: {}", len(study.trials))

	logger.info("Best 5 trials:")
	trials = study.trials
	trials = sorted(trials, key=lambda elem: elem.value, reverse=True)[:5]

	for i, trial in enumerate(trials):
		logger.info("Trial {}", i)
		logger.info("\tValue: {:.2f}", trial.value)

		logger.info("\tParams: ")
		for key, value in trial.params.items():
			logger.info("\t\t{}: {}", key, value)


def main():
	dataPath = r"D:\data\Research\BCI_dataset\NewData"
	optunaFile = r"D:\research\EEGNet\Data\Experiments\Optuna\optuna.log"

	logger.add(optunaFile, level="INFO")
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)

	patients = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]

	for pat in patients:
		loader.loadMatlab(
			path=os.path.join(dataPath, str(pat)),
			sourceSR=500,
			targetSR=323,
			windows=[(0.2, 0.5)],
			baselineWindow=(0.2, 0.3),
			shuffle=True,
			store=True,
			name=pat
		)

	loader.saveHDF(
		data=loader.stored,
		dirpath=dataPath,
		filename="All_patients_sr323"
	)

	for pat, value in loader.stored.items():
		logger.debug("")

		data = value["data"]
		labels = value["labels"]

		data = np.expand_dims(data, axis=1)

		dataset = splitDataset(data, labels, trainPart=0.8, valPart=0.1, permutation=True, seed=42069)

		config = {
			"train_config": {
				"epochs": 500,
				"batchsize": 64
			},
			"paths": {
				"checkpoint": "./Data/Experiments/Optuna/%d" % pat,
				"logs": "./Data/Experiments/Optuna/%d/Logs" % pat
			}
		}

		optunaTrainer = OptunaTrainer(config=config, dataset=dataset)

		study = optuna.create_study(direction="maximize")

		try:
			study.optimize(optunaTrainer, n_trials=700, show_progress_bar=True)

			logger.info("Optuna train has been finished for patient #{}", pat)
			studyInfo(study)
		except Exception as e:
			logger.error("Optuna train has been failed for patient #{} with error: {}", pat, e)


if __name__ == "__main__":
	jointTrainOptuna()