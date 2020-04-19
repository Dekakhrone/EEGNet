import os
import yaml
import datetime

import optuna
import numpy as np
import tensorflow as tf
from loguru import logger
from functools import partial
from tensorflow.keras.callbacks import ModelCheckpoint

from Model import EEGNet
from Utils.DataLoader import DataHandler, Formats, splitDataset, crossValGenerator
from Utils.Augmentations import getAugmenter
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


def test(model, checkpointPath, dataset, **kwargs):
	model.load_weights(checkpointPath)

	data, labels = dataset

	pred = model.predict(data)[:, 1]
	auc = ROC(labels, pred, **kwargs)

	return auc


def train(model, dataset, weigthsPath, epochs=100, batchsize=128, crossVal=False, verbose=0, augmenter=None):
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
			augmenter=augmenter
		)

		auc.append(test(model, checkpointPath, testSet))

		tf.keras.backend.clear_session()

	return np.mean(auc)


def _train(model, dataset, weigthsPath, epochs=100, batchsize=128, verbose=0, augmenter=None):
	os.makedirs(weigthsPath, exist_ok=True)

	trainset = dataset["train"]
	valset = dataset["val"]

	if augmenter is not None:
		augmented = [augmenter(*trainset) for _ in range(epochs)]

		trainset = (
			np.concatenate([data[0] for data in augmented], axis=0),
			np.concatenate([data[1] for data in augmented], axis=0)
		)

	checkpointPath = os.path.join(weigthsPath, "best.h5")
	checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=verbose, save_best_only=True)

	model.fit(*trainset, batch_size=batchsize, epochs=epochs, verbose=verbose, validation_data=valset,
	          callbacks=[checkpointer])

	return checkpointPath


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


def separateTrain():
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


def jointTrainOptuna():
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)
	patients = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
	patients = [str(elem) for elem in patients]

	logger.add(
		sink="./Data/Experiments/Optuna/optuna.log",
		level="INFO"
	)

	dataset = loader.loadHDF(
		filepath=r"D:\data\Research\BCI_dataset\NewData\All_patients_sr323.hdf",
		keys=patients
	)

	trainSet = {}
	testSet = {}
	for key, value in dataset.items():
		data = value["data"]
		labels = value["labels"]

		data = np.expand_dims(data, axis=1)
		dataset = splitDataset(data, labels, trainPart=0.8, valPart=0.0, permutation=True, seed=42069)

		trainSet[key] = dataset["train"]
		testSet[key] = dataset["test"]

	optunaTrainer = OptunaTrainer(
		checkpointPath="./Data/Experiments/Optuna",
		batchsize=64,
		epochs=200
	)

	study = optuna.create_study(direction="maximize")

	trainer = partial(optunaTrainer, dataset=trainSet, crossVal=False)

	study.optimize(trainer, n_trials=250, show_progress_bar=True)
	studyInfo(
		study,
		file="./Data/Experiments/Optuna/optuna_results.csv"
	)


def customParamsTrain():
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)
	patients = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
	patients = [str(elem) for elem in patients]

	date = str(datetime.date.today())

	logger.add(
		sink="./Data/Experiments/Custom/{}.log".format(date),
		level="INFO"
	)

	epochs = 250
	batchsize = 16
	learningRate= 1e-3
	temporalLength = 40
	dropoutRate=0.3
	D = 3
	poolKernel = 16

	logger.info("Model and train parameters: epochs {}, batchsize {}, learningRate {}, temporalLength {}, "
	            "dropoutRate {}, D {}, poolKernel {}".
	            format(epochs, batchsize, learningRate, temporalLength, dropoutRate, D, poolKernel))

	dataset = loader.loadHDF(
		filepath=r"D:\data\Research\BCI_dataset\NewData\All_patients_sr323.hdf",
		keys=patients
	)

	trainSet = {}
	testSet = {}
	for key, value in dataset.items():
		data = value["data"]
		labels = value["labels"]

		data = np.expand_dims(data, axis=1)
		dataset = splitDataset(data, labels, trainPart=0.8, valPart=0.0, permutation=True, seed=42069)

		trainSet[key] = dataset["train"]
		testSet[key] = dataset["test"]

	augmenter = getAugmenter()

	for key, value in trainSet.items():
		patientPath = "./Data/Experiments/Custom/{}/{}_patient".format(date, key)
		shape = value[0].shape[-2:]

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

		model.compile(
			loss="sparse_categorical_crossentropy",
			optimizer=tf.optimizers.Adam(learning_rate=learningRate, decay=learningRate / epochs),
			metrics=["accuracy"]
		)

		valAUC = train(
			model=model,
			dataset=value,
			weigthsPath=patientPath,
			epochs=epochs,
			batchsize=batchsize,
			crossVal=False,
			verbose=2,
			augmenter=augmenter
		)

		testAUC = test(
			model=model,
			checkpointPath=os.path.join(patientPath, "best.h5"),
			dataset=testSet[key],
			wpath=patientPath
		)

		logger.info("Patient #{}: val auc {:.2f}, test auc {:.2f}".format(key, valAUC, testAUC))


if __name__ == "__main__":
	customParamsTrain()