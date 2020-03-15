import os
from enum import Enum
from random import shuffle, seed

import numpy as np
from scipy.io import loadmat
from mne.filter import resample


EEG_SAMPLE_RATE = 500  # Hz


class Formats(str, Enum):
	ttc = "ttc" # trials, time, channels
	tct = "tct" # trials, channels, time


axes = {
	Formats.ttc: (2, 0, 1),
	Formats.tct: (2, 1, 0)
}


def permutate(*arr, saveOrder=False, seedValue=1234):
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
	factor = targetSR / sourceSR

	if factor == 1:
		return data

	return resample(data, up=max(1., factor), down=max(1., 1 / factor), npad="auto", axis=axis)


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

		if store:
			if name is None:
				name = path

			self.stored[name] = {
				"data": data,
				"labels": labels,
				"format": self.format,
				"sample_rate": currentSR
			}

		return data, labels


	def loadHDF(self, path, store=False):
		pass


	def loadNumpy(self, path, store=False):
		pass


	def saveHDF(self, data, path):
		pass


	def saveNumpy(self, data, path):
		pass


def main():
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.ttc)

	data = loader.loadMatlab(r'D:\data\Research\BCI_dataset\NewData\25', sourceSR=500, targetSR=323,
	                             windows=[(0.2, 0.5)], baselineWindow=(0.2, 0.3), store=True, name="25")

	print()

if __name__ == '__main__':
	main()