import numpy as np

from Utils.DataLoader import permutate, DataHandler
from AudioAugmentation import Augmentations as augs
from AudioAugmentation.Core.Base import Sequential, NormalizationTypes, AugTypes


class EEGAugmenter(Sequential):
	def __init__(self, augmenters, randomOrder=False, returnSpectrogram=False, normalization=NormalizationTypes.origin,
	             typeOrder=(AugTypes.audio, AugTypes.spectrogram), sampleRate=16000, oversample=False, **kwargs):

		super().__init__(augmenters, randomOrder, returnSpectrogram, normalization, typeOrder, sampleRate, **kwargs)
		self.oversample = oversample


	def __call__(self, signals, labels):
		posIdxs = np.argwhere(labels == 1)
		negIdxs = np.argwhere(labels == 0)

		oversample = posIdxs if posIdxs.size < negIdxs.size else negIdxs
		diff = abs(posIdxs.size - negIdxs.size) if self.oversample else 0

		shape = signals.shape if not self.oversample else (2 * (oversample.size + diff), ) + signals.shape[1:]
		newSignals = np.empty(shape, dtype=signals.dtype)
		newLabels = np.empty(shape[0], dtype=labels.dtype)

		for t, trial in enumerate(signals):
			trial = trial[0]
			for c, channel in enumerate(trial):
				newSignals[t, 0, c] = self.apply(channel)

			newLabels[t] = labels[t]

		oversampleIdxs = np.random.choice(np.ravel(oversample), diff)

		for i, idx in enumerate(oversampleIdxs):
			trial = signals[idx, 0]
			for c, channel in enumerate(trial):
				newSignals[len(signals) + i, 0, c] = self.apply(channel)

			newLabels[len(signals) + i] = labels[idx]

		newSignals = permutate(newSignals, saveOrder=True)
		newLabels = permutate(newLabels, saveOrder=True)

		return newSignals, newLabels


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