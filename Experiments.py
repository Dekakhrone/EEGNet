import os
import datetime

import optuna
import numpy as np
import tensorflow as tf
from loguru import logger
from functools import partial

import config
from Model import EEGNet
from Optuna import OptunaTrainer, studyInfo
from Train import test, train
from Utils.DataLoader import DataHandler, Formats, splitDataset
from Utils.Augmentations import getAugmenter, getOversampler


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


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

		dataset = splitDataset(data, labels, trainPart=0.8, valPart=0.1, permutation=True, seedValue=42069)

		optunaTrainer = OptunaTrainer(
			checkpointPath="./Data/Experiments/Optuna/%d" % pat,
			epochs=500,
			batchsize=64,
			logPath="./Data/Experiments/Optuna/%d/Logs" % pat
		)

		study = optuna.create_study(direction="maximize")
		trainer = partial(optunaTrainer, dataset=dataset, crossVal=False)

		try:
			study.optimize(trainer, n_trials=700, show_progress_bar=True)

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
		dataset = splitDataset(data, labels, trainPart=0.8, valPart=0.0, permutation=True, seedValue=42069)

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
	time = datetime.datetime.now().time()
	time = "{}-{}".format(time.hour, time.minute)

	experimentFolder = "./Data/Experiments/{}/{}".format(date, time)

	logger.add(
		sink=os.path.join(experimentFolder, ".log"),
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
		filepath=r"D:\data\Research\BCI_dataset\NewData\All_patients_sr323_ext_win.hdf",
		keys=patients
	)

	oversampler = getOversampler()

	trainSet = {}
	testSet = {}
	for key, value in dataset.items():
		data = value["data"]
		labels = value["labels"]

		data = np.expand_dims(data, axis=1)
		data, labels = oversampler.oversample(data, labels, shuffle=True)

		dataset = splitDataset(data, labels, trainPart=0.8, valPart=0.0, permutation=True, seedValue=42069)

		trainSet[key] = dataset["train"]
		testSet[key] = dataset["test"]

	augmenter = getAugmenter()

	for key, value in trainSet.items():
		patientPath = os.path.join(experimentFolder, "{}_patient".format(key))
		shape = list(value[0].shape[-2:])

		shape[1] = int(config.window[1] * config.sampleRate) - int(config.window[0] * config.sampleRate)

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
			augmenter=augmenter,
			augProb=0.9,
			oversample=False,
			clip=(0.05, 0.35)
		)

		testAUC = test(
			model=model,
			checkpointPath=os.path.join(patientPath, "best.h5"),
			dataset=testSet[key],
			wpath=patientPath,
			clip=(0.05, 0.35)
		)

		logger.info("Patient #{}: val auc {:.2f}, test auc {:.2f}".format(key, valAUC, testAUC))


if __name__ == "__main__":
	customParamsTrain()