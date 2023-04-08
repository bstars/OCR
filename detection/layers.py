import torch
from torch import nn

class ConvNormRelu(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(ConvNormRelu, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.1)
		)

	def forward(self,x):
		return self.net(x)

class YoloConvBlockNaive(nn.Module):
	"""
	Gives 2 outputs
		1. A route with same channels as input
		2. A route with twice the channels as input
	"""
	def __init__(self, in_channels, filters, return_route=True):
		super(YoloConvBlockNaive, self).__init__()
		self.return_route = return_route
		self.net1 = nn.Sequential(
			ConvNormRelu(in_channels, filters, 1, 1, 0),
			ConvNormRelu(filters, filters*2, 3, 1, 1),
			ConvNormRelu(filters*2, filters, 1, 1, 0),
			ConvNormRelu(filters, filters*2, 3, 1, 1),
			ConvNormRelu(filters*2, filters, 1, 1, 0)
		)

		self.net2 = ConvNormRelu(filters, filters*2, 3, 1, 1)

	def forward(self, x):
		route = self.net1(x)
		out = self.net2(route)
		if self.return_route:
			return route, out
		else:
			return out

class YoloDetectionBlockNaive(nn.Module):
	def __init__(self, in_channels):
		super(YoloDetectionBlockNaive, self).__init__()
		self.net = nn.Conv2d(in_channels, 5, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		out = self.net(x)
		return out

class FeatureFusionNaive(nn.Module):
	"""
	Fusion low-resolution feature with high-resolution by
	upsampling the low-resolution feature and concatenating
	with high-resolution feature along channel dimension
	"""
	def __init__(self, low_res_channels, scale_factor=2):
		super(FeatureFusionNaive, self).__init__()
		self.conv = ConvNormRelu(low_res_channels, low_res_channels//2, 1, 1, 0)
		self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)

	def forward(self, low_res_feature, high_res_feature):
		low_res_feature = self.conv(low_res_feature)
		low_res_feature = self.upsample(low_res_feature)
		# print(low_res_feature.shape, high_res_feature.shape)
		out = torch.cat([low_res_feature, high_res_feature], dim=1)
		return out

class UpSampling(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
		super(UpSampling, self).__init__()

		self.net = nn.Sequential(
			nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding),
			nn.LeakyReLU(0.1),
			ConvNormRelu(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		return self.net(x)

if __name__ == '__main__':
	x = torch.randn(1, 64, 48, 48)
	net = UpSampling(64, 128, 2, 2, 0)
	y = net(x)
	print(y.shape)