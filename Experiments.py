import os
from random import randint

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from Utils.DataLoader import DataHandler, Formats, permutate
from Model import EEGNet


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def splitDataset(data, labels, trainPart=0.8, valPart=0.2, permutation=True):
	assert labels.shape[0] == data.shape[0]
	assert trainPart + valPart <= 1

	length = labels.shape[0]
	testPart = 1.0 - trainPart - valPart

	partList = ["train", "val", "test"]
	parts = {"train": trainPart, "val": valPart, "test": testPart}

	dataset = {}
	start = 0
	for part in partList:
		ratio = parts.get(part)
		end = start + round(ratio * length)

		seed = randint(0, 2**16)

		dataset[part] = (
			data[start:end] if not permutation else permutate(data[start:end], saveOrder=True, seedValue=seed),
			labels[start:end] if not permutation else permutate(labels[start:end], saveOrder=True, seedValue=seed)
		)

		start = end

	return dataset


def train(model, dataset, weigthsPath, logPath, trainPart=0.8, validPart=0.2, epochs=100, batchsize=128):
	for path in [weigthsPath, logPath]:
		os.makedirs(path, exist_ok=True)

	checkpointPath = os.path.join(weigthsPath, "best.h5")
	checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=1, save_best_only=True)

	dataset = splitDataset(*dataset, trainPart=trainPart, valPart=validPart, permutation=True)

	# dataTrain, labelsTrain = dataset["train"]

	model.fit(*dataset["train"], batch_size=batchsize, epochs=epochs, verbose=2, validation_data=dataset["val"],
	          callbacks=[checkpointer])

	model.load_weights(checkpointPath)

	dataTest, labelsTest = dataset["test"]

	probs = model.predict(dataTest)
	preds = probs.argmax(axis=-1)
	acc = np.mean(preds == labelsTest)
	print("Classification accuracy: %f " % (acc))


def main():
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)
	data, labels = loader.loadMatlab(
		path=r'D:\data\Research\BCI_dataset\NewData\25',
		sourceSR=500,
		targetSR=323,
		windows=[(0.2, 0.5)],
		baselineWindow=(0.2, 0.3),
		shuffle=True
	)

	shape = data.shape
	data = np.expand_dims(data, axis=1)

	model = EEGNet(
		categoriesN=2,
		electrodes=shape[1],
		samples=shape[2]
	)

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	train(
		model=model,
		dataset=(data, labels),
		weigthsPath="./Data/Experiments",
		logPath="./Data/Experiments/Logs",
		trainPart=0.7,
		validPart=0.1,
		epochs=100,
		batchsize=128
	)


if __name__ == "__main__":
	main()