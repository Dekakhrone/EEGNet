import os
from enum import Enum
from copy import deepcopy
from random import shuffle, seed, randint

import h5py
import numpy as np
from scipy.io import loadmat
from mne.filter import resample
from matplotlib import pyplot as plt
from math import sqrt


EEG_SAMPLE_RATE = 500  # Hz


class Formats(str, Enum):
	ttc = "ttc" # trials, time, channels
	tct = "tct" # trials, channels, time


axes = {
	Formats.ttc: (2, 0, 1),
	Formats.tct: (2, 1, 0)
}


def splitDataset(data, labels, trainPart=0.8, valPart=0.2, permutation=True, seedValue=None):
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

		seedValue = randint(0, 2 ** 16) if seedValue is None else seedValue

		dataset[part] = (
			data[start:end] if not permutation else permutate(data[start:end], saveOrder=True, seedValue=seedValue),
			labels[start:end] if not permutation else permutate(labels[start:end], saveOrder=True, seedValue=seedValue)
		)

		start = end

	return dataset


def crossValGenerator(data, labels, trainPart=0.8, valPart=0.2, permutation=True, seedValue=None):
	assert labels.shape[0] == data.shape[0]
	assert trainPart + valPart <= 1

	seedValue = randint(0, 2 ** 16) if seedValue is None else seedValue

	length = labels.shape[0]
	valSamples = int(length * valPart)
	repeats = int(length / valSamples)

	idxs = [(i * valSamples, (i + 1) * valSamples) for i in range(repeats)]

	for start, end in idxs:
		valData = data[start:end]
		valLabels = labels[start:end]

		trainData = np.concatenate((data[:start], data[end:]), axis=0)
		trainLabels = np.concatenate((labels[:start], labels[end:]), axis=0)

		dataset = {
			"train": (
				trainData if not permutation else permutate(trainData, saveOrder=True, seedValue=seedValue),
				trainLabels if not permutation else permutate(trainLabels, saveOrder=True, seedValue=seedValue)
			),
			"val": (
				valData if not permutation else permutate(valData, saveOrder=True, seedValue=seedValue),
				valLabels if not permutation else permutate(valLabels, saveOrder=True, seedValue=seedValue)
			)
		}

		yield dataset


def permutate(arr, saveOrder=False, seedValue=1234):
	idxs = list(range(len(arr)))

	if saveOrder:
		seed(seedValue)

	shuffle(idxs)

	if isinstance(arr, np.ndarray):
		arr = arr[idxs]
	elif isinstance(arr, list):
		arr = [arr[idx] for idx in idxs]
	else:
		raise TypeError

	return arr


def _resample(data, sourceSR, targetSR, axesFormat=Formats.tct):
	axis = axes[axesFormat].index(0)
	_factor = targetSR / sourceSR

	if _factor == 1:
		return data

	return resample(data, up=max(1., _factor), down=max(1., 1 / _factor), npad="auto", axis=axis)


class DataHandler:
	def __init__(self, epochs:tuple, dformat=Formats.tct):
		self.epochs = epochs

		self.format = dformat

		self.stored = {}


	@staticmethod
	def _samples2secs(samples, sampleRate):
		return round(samples / sampleRate, 3)


	@staticmethod
	def _secs2samples(secs, sampleRate):
		return int(secs * sampleRate)


	def _baselineNorm(self, data, window, sampleRate):
		axis = axes[self.format].index(0)

		start = self._secs2samples(window[0] - self.epochs[0], sampleRate)
		end = self._secs2samples(window[1] - self.epochs[0], sampleRate)

		slc = tuple(slice(0, s) if i != axis else slice(start, end) for i, s in enumerate(data.shape))
		baseline = np.expand_dims(data[slc].mean(axis=axis), axis=axis)

		data -= baseline

		return data


	def loadMatlab(self, path, sourceSR, targetSR=None, windows=None, baselineWindow=None, shuffle=False, store=False,
	               name=None):
		container = {}

		if os.path.isdir(path):
			positive = loadmat(os.path.join(path, "eegT.mat"))["eegT"]
			negative = loadmat(os.path.join(path, "eegNT.mat"))["eegNT"]

			labels = np.hstack((np.ones(positive.shape[2]), np.zeros(negative.shape[2])))
			data = np.concatenate((positive, negative), axis=-1)
		elif os.path.isfile(path):
			raise ValueError

		data = data.transpose(*axes[self.format])

		if baselineWindow is not None:
			data = self._baselineNorm(data, baselineWindow, sourceSR)

		if targetSR is not None and targetSR != sourceSR:
			currentSR = targetSR
			data = _resample(data, sourceSR, targetSR, self.format)
		else:
			currentSR = sourceSR

		if windows is not None:
			timeIndicies = []

			for win in windows:
				start = self._secs2samples(win[0] - self.epochs[0], currentSR)
				end = self._secs2samples(win[1] - self.epochs[0], currentSR)

				timeIndicies.extend(list(range(start, end)))

			data = data[:, :, timeIndicies] if self.format == Formats.tct else data[:, timeIndicies, :]

		if shuffle:
			data = permutate(data, saveOrder=True)
			labels = permutate(labels, saveOrder=True)

		if name is None:
			name = path

		container[name] = {
			"data": data,
			"labels": labels,
			"format": self.format.value,
			"sample_rate": currentSR
		}

		if store:
			self.stored[name] = deepcopy(container[name])

		return container


	def loadHDF(self, filepath, store=False, keys=None):
		file = h5py.File(filepath, "r")

		container = {}
		try:
			for name in file.keys():
				keys = file.keys() if keys is None else keys
				if name not in keys:
					continue

				group = file[name]

				container[name] = {
					"data": group["data"][:],
					"labels": group["labels"][:],
					"format": group.attrs.get("format"),
					"sample_rate": group.attrs.get("sample_rate")
				}
		finally:
			file.close()

		if store:
			self.stored = deepcopy(container)

		return container


	def saveHDF(self, data, dirpath, filename, meta=True):
		os.makedirs(dirpath, exist_ok=True)

		filename = os.path.splitext(filename)[0] + ".hdf"
		filepath = os.path.join(dirpath, filename)

		file = h5py.File(filepath, "w")

		if isinstance(data, dict):
			for name, ndata in data.items():
				name = str(name)
				group = file.create_group(name)

				for attr, value in ndata.items():
					if attr in ["format", "sample_rate"] and meta:
						group.attrs[attr] = value
						continue

					group.create_dataset(attr, data=value)
		elif isinstance(data, tuple):
			data, labels = data

			file.create_dataset("data", data=data)
			file.create_dataset("labels", data=labels)
		elif isinstance(data, np.ndarray):
			file.create_dataset("data", data=data)
		else:
			raise TypeError

		file.close()
		print("Data has been written to {}".format(filepath))


def factor(number):
	number = number if not number % 2 else number + 1
	f = [(d, number // d) for d in range(2, int(sqrt(number) + 1)) if not number % d]

	return f[-1]


def plot(data, labels, wpath, name=None, samples=10):
	plt.ioff()
	data = np.squeeze(data)

	trials, channels, time = data.shape
	rows, cols = factor(channels)
	time = list(range(time))

	os.makedirs(wpath, exist_ok=True)

	for i, trial in enumerate(data):
		label = labels[i]

		fig = plt.figure(figsize=(16, 8))
		plt.title("Trial #{}, label {}".format(i, label))

		axs = fig.subplots(rows, cols)
		fig.subplots_adjust(hspace=.5, wspace=.001)

		axs = axs.ravel()

		for j, electrode in enumerate(trial):
			axs[j].grid(which="both")
			axs[j].plot(time, electrode, "b", label="channel {}".format(j))

		if name is not None:
			_name, _ = os.path.splitext(name)
			_name = "{}-trial_{}.png".format(_name, i)
		else:
			_name = "trial_{}.png".format(i)

		plt.savefig(os.path.join(wpath, _name))

		if i == samples:
			break


def main():
	path = r'D:\data\Research\BCI_dataset\NewData'
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)

	patients = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]

	for pat in patients:
		print("\rPatient #{} is being processed".format(pat), end="")
		loader.loadMatlab(
			path=os.path.join(path, str(pat)),
			sourceSR=500,
			targetSR=323,
			windows=[(0.15, 0.55)],
			baselineWindow=(0.2, 0.3),
			shuffle=True,
			store=True,
			name=pat
		)

	filename = "All_patients_sr323_ext_win.hdf"
	loader.saveHDF(
		data=loader.stored,
		dirpath=path,
		filename=filename
	)

	loader.stored = {}
	loader.loadHDF(os.path.join(path, filename), store=True)


def testPlot():
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)

	dataset = loader.loadMatlab(
		path=r"D:\data\Research\BCI_dataset\NewData\25",
		sourceSR=EEG_SAMPLE_RATE,
		windows=[(0.2, 0.5)],
		baselineWindow=(0.2, 0.3),
		name="25"
	)

	# dataset = loader.loadHDF(
	# 	filepath=r"D:\data\Research\BCI_dataset\NewData\All_patients_sr323.hdf",
	# 	keys="25"
	# )

	data = dataset["25"]["data"]
	labels = dataset["25"]["labels"]

	plot(data, labels, "../Data/Plots")


if __name__ == '__main__':
	main()