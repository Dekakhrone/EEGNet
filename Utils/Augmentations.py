import numpy as np

from Utils.DataLoader import permutate, DataHandler, Formats, plot
from AudioAugmentation import Augmentations as augs
from AudioAugmentation.Core.Base import Sequential
from AudioAugmentation.Core.Types import NormalizationTypes, AugTypes, Colors


class EEGAugmenter(Sequential):
	def __init__(self, augmenters, randomOrder=False, returnSpectrogram=False, normalization=NormalizationTypes.origin,
	             typeOrder=(AugTypes.audio, AugTypes.spectrogram), sampleRate=16000, clip=(None, None), **kwargs):

		super().__init__(augmenters, randomOrder, returnSpectrogram, normalization, typeOrder, sampleRate, **kwargs)
		self.clip = clip


	def __call__(self, data, labels, oversample=False, shuffle=False):
		oversample, diff = getIdxToOversample(labels)
		n = 2 if oversample else 1

		shape = (2 * oversample.size + n * diff, ) + data.shape[1:]
		newData = np.empty(shape, dtype=data.dtype)
		newLabels = np.empty(shape[0], dtype=labels.dtype)

		for t, trial in enumerate(data):
			trial = trial[0]
			for c, channel in enumerate(trial):
				newData[t, 0, c] = self.apply(channel)

			newLabels[t] = labels[t]

		oversampleIdxs = np.random.choice(np.ravel(oversample), oversample * diff)

		for i, idx in enumerate(oversampleIdxs):
			trial = data[idx, 0]
			for c, channel in enumerate(trial):
				newData[len(data) + i, 0, c] = self.apply(channel)

			newLabels[len(data) + i] = labels[idx]

		if shuffle:
			newData = permutate(newData, saveOrder=True)
			newLabels = permutate(newLabels, saveOrder=True)

		_min = int((self.clip[0] or 0) * self.sampleRate)
		_max = int((self.clip[1] or len(data) // self.sampleRate) * self.sampleRate)
		newData = newData[..., _min:_max]

		return newData, newLabels


def getIdxToOversample(labels):
	posIdxs = np.argwhere(labels == 1)
	negIdxs = np.argwhere(labels == 0)

	oversample = posIdxs if posIdxs.size < negIdxs.size else negIdxs
	diff = abs(posIdxs.size - negIdxs.size)

	return oversample, diff


def getAugmenter():
	seqKwargs = {
		"normalization": NormalizationTypes.none,
	    "typeOrder": (AugTypes.audio, ),
	    "sampleRate": 323,
		"clip": (0.05, 0.35)
	}

	aug = EEGAugmenter(
		[
			augs.LoudnessAug(loudnessFactor=(0.75, 1.25)),
			augs.ShiftAug(0.03),
			augs.SyntheticNoiseAug(noiseColor=Colors.white, noiseFactor=(0.25, 0.5))
		],
		**seqKwargs
	)

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