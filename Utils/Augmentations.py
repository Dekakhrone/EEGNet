import numpy as np

from Utils.DataLoader import permutate, DataHandler, Formats, plot
from AudioAugmentation import Augmentations as augs
from AudioAugmentation.Core.Base import Sequential
from AudioAugmentation.Core.Types import NormalizationTypes, AugTypes, Colors


class EEGAugmenter(Sequential):
	def __init__(self, augmenters, randomOrder=False, returnSpectrogram=False, normalization=NormalizationTypes.origin,
	             typeOrder=(AugTypes.audio, AugTypes.spectrogram), sampleRate=16000, oversample=False, **kwargs):

		super().__init__(augmenters, randomOrder, returnSpectrogram, normalization, typeOrder, sampleRate, **kwargs)
		self.oversample = oversample


	def __call__(self, data, labels, shuffle=False):
		posIdxs = np.argwhere(labels == 1)
		negIdxs = np.argwhere(labels == 0)

		oversample = posIdxs if posIdxs.size < negIdxs.size else negIdxs
		diff = abs(posIdxs.size - negIdxs.size) if self.oversample else 0

		shape = data.shape if not self.oversample else (2 * (oversample.size + diff),) + data.shape[1:]
		newData = np.empty(shape, dtype=data.dtype)
		newLabels = np.empty(shape[0], dtype=labels.dtype)

		for t, trial in enumerate(data):
			trial = trial[0]
			for c, channel in enumerate(trial):
				newData[t, 0, c] = self.apply(channel)

			newLabels[t] = labels[t]

		oversampleIdxs = np.random.choice(np.ravel(oversample), diff)

		for i, idx in enumerate(oversampleIdxs):
			trial = data[idx, 0]
			for c, channel in enumerate(trial):
				newData[len(data) + i, 0, c] = self.apply(channel)

			newLabels[len(data) + i] = labels[idx]

		if shuffle:
			newData = permutate(newData, saveOrder=True)
			newLabels = permutate(newLabels, saveOrder=True)

		return newData, newLabels


def getAugmenter():
	seqKwargs = {
		"normalization": NormalizationTypes.none,
	    "typeOrder": (AugTypes.audio, ),
	    "sampleRate": 323,
	    "oversample": True
	}

	aug = EEGAugmenter(
		[
			augs.LoudnessAug(loudnessFactor=(0.75, 1.25)),
			augs.ShiftAug(0.03),
			augs.SyntheticNoiseAug(noiseColor=Colors.white, noiseFactor=(0.25, 0.5)),
		],
		**seqKwargs
	)

	return aug


def main():
	from AudioAugmentation.Augmentations import ShiftAug

	aug = EEGAugmenter([ShiftAug(0.1)], oversample=True, sampleRate=323, normalization=NormalizationTypes.none)

	data = np.random.randint(1, 100, (100, 1, 19, 93)).astype(np.float32)
	labels = np.random.randint(0, 2, 100).astype(np.int32)

	aug(data, labels)


def visualizeAugmentations():
	pass


if __name__ == "__main__":
	main()