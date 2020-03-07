from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
	Conv2D, AvgPool2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, Dense, Activation, Dropout, Flatten
)
from keras.constraints import max_norm


class EEGNet(Sequential):
	def __init__(self, categoriesN, electrodes=64, samples=128, dropoutRate=0.5, F1=8, D=2, F2=16, normRate=0.25):
		super().__init__()

		if samples % 4 != 0:
			raise ValueError("The number of samples must be a multiple of 4")

		temporalLength = samples // 2
		avgKernel = (4 * samples) // 128

		layers = [
			Conv2D(filters=F1, kernel_size=(1, temporalLength), padding="same", use_bias=False, name="conv_0"),
			BatchNormalization(axis=1, name="bn_0"),

			DepthwiseConv2D(kernel_size=(electrodes, 1), depth_multiplier=D, depthwise_constraint=max_norm(1.),
			                name="dp_conv_0"),
			BatchNormalization(axis=1, name="bn_1"),
			Activation(activation="elu", name="act_0"),
			AvgPool2D(pool_size=(1, avgKernel), name="avg_pool_0"),
			Dropout(dropoutRate, name="dropout_0"),

			SeparableConv2D(filters=F2, kernel_size=(1, temporalLength // 4), padding="same", use_bias=False,
			                name="spr_conv_0"),
			BatchNormalization(axis=1, name="bn_2"),
			Activation(activation="elu", name="act_1"),
			AvgPool2D(pool_size=(1, avgKernel * 2), name="avg_pool_1"),
			Dropout(dropoutRate, name="dropout_1"),

			Flatten(name="flatten"),
			Dense(categoriesN, name="dense", kernel_constraint=max_norm(normRate)),
			Activation(activation="softmax", name="softmax")
		]

		for layer in layers:
			self.add(layer)


def main():
	model = EEGNet(10)


if __name__ == "__main__":
	main()