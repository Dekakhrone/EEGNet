import numpy as np

import config
from Utils.DataLoader import permutate, DataHandler, Formats, plot
from AudioAugmentation import Augmentations as augs
from AudioAugmentation.Core.Base import Sequential, AudioAugmenter
from AudioAugmentation.Core.Types import NormalizationTypes, AugTypes, Colors


class EEGAugmenter(Sequential):
	def __call__(self, data, labels, oversample=False, shuffle=False, clip=(None, None)):
		newData = np.empty(data.shape, dtype=data.dtype)
		newLabels = np.empty(labels.shape[0], dtype=labels.dtype)

		for t, trial in enumerate(data):
			trial = trial[0]
			for c, channel in enumerate(trial):
				newData[t, 0, c] = self.apply(channel)

			newLabels[t] = labels[t]

		if oversample:
			scarce, _ = getIdxToOversample(labels)
			newData, newLabels = self.oversample(newData, newLabels, shuffle, clip, determined=scarce)

		if shuffle:
			newData = permutate(newData, saveOrder=True)
			newLabels = permutate(newLabels, saveOrder=True)

		newData = clipAxis(newData, clip, self.sampleRate, axis=3)

		return newData, newLabels


	def oversample(self, data, labels, shuffle=False, clip=(None, None), determined=None):
		scarce, diff = getIdxToOversample(labels)
		scarce = scarce if determined is None else determined

		if len(scarce) == 0:
			return data, labels

		oversampleIdxs = np.random.choice(scarce, diff)

		shape = (diff, ) + data.shape[1:]

		overData = np.empty(shape, dtype=data.dtype)
		overLabels = np.empty(shape[0], dtype=labels.dtype)

		for i, idx in enumerate(oversampleIdxs):
			trial = data[idx, 0]
			for c, channel in enumerate(trial):
				overData[i, 0, c] = self.apply(channel)

			overLabels[i] = labels[idx]

		data = np.concatenate((data, overData), axis=0)
		labels = np.concatenate((labels, overLabels), axis=0)

		if shuffle:
			data = permutate(data, saveOrder=True)
			labels = permutate(labels, saveOrder=True)

		data = clipAxis(data, clip, self.sampleRate, axis=3)

		return data, labels


class Sin(AudioAugmenter):
	def __init__(self, name=None, factor=(0.5, 2.0)):
		super().__init__(name)

		self.factor = factor


	def _getSin(self, size):
		X = np.linspace(-3, 3, size, dtype=np.float32).reshape(-1, 1)

		a = np.random.choice(np.arange(10), 1)[0]
		p = np.random.exponential(4, 1)[0]

		def func(x):
			from math import sin
			return a * sin(p * x)

		f = np.vectorize(func)
		return f(X)


	def apply(self, data, sampleRate, returnSpectrogram=False, **kwargs):
		self._validateData(data)
		multiplier = self._setParam(self.factor)

		sin = np.ravel(self._getSin(data.size))
		data += sin * multiplier

		return data


def clipAxis(data, borders, sampleRate, axis=0):
	axis = np.ndim(data) + axis if axis < 0 else axis

	_min = int((borders[0] or 0) * sampleRate)
	_max = int((borders[1] or data.shape[axis] / sampleRate) * sampleRate)

	slices = tuple(slice(0, dim) if i != axis else slice(_min, _max) for i, dim in enumerate(data.shape))

	return data[slices]


def getIdxToOversample(labels):
	posIdxs = np.argwhere(labels == 1)
	negIdxs = np.argwhere(labels == 0)

	scarce = posIdxs if posIdxs.size < negIdxs.size else negIdxs
	diff = abs(posIdxs.size - negIdxs.size)

	return np.ravel(scarce), diff


def getAugmenter():
	seqKwargs = {
		"normalization": NormalizationTypes.none,
	    "typeOrder": (AugTypes.audio, ),
	    "sampleRate": config.sampleRate
	}

	aug = EEGAugmenter(
		[
			# augs.LoudnessAug(loudnessFactor=(0.75, 1.25)),
			Sin(factor=(0.5, 0.8)),
			augs.ShiftAug(shiftFactor=(0.01, 0.03))
			# augs.SyntheticNoiseAug(noiseColor=Colors.white, noiseFactor=(0.25, 0.5)),
		],
		**seqKwargs
	)

	return aug


def getOversampler():
	seqKwargs = {
		"normalization": NormalizationTypes.none,
		"typeOrder": (AugTypes.pseudo, ),
		"sampleRate": config.sampleRate
	}

	aug = EEGAugmenter(augmenters=[augs.Identity()], **seqKwargs)

	return aug


def main():
	aug = getAugmenter()

	data = np.random.randint(1, 100, (100, 1, 19, 93)).astype(np.float32)
	labels = np.random.randint(0, 2, 100).astype(np.int32)

	aug(data, labels)


def visualizeAugmentations():
	loader = DataHandler(epochs=(-0.5, 1), dformat=Formats.tct)

	dataset = loader.loadHDF(
		filepath=r"D:\data\Research\BCI_dataset\NewData\All_patients_sr323.hdf",
		keys="25"
	)

	data = dataset["25"]["data"]
	labels = dataset["25"]["labels"]

	plot(data, labels, "../Data/Plots/Clean", samples=3)

	aug = getAugmenter()

	data, labels = aug(data[:, np.newaxis, ...], labels)

	plot(np.squeeze(data), labels, "../Data/Plots/Augmented", samples=3)


if __name__ == "__main__":
	visualizeAugmentations()