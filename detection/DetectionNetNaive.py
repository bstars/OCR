import torch
from torch import nn
import numpy as np
import abc

from torchvision.models import resnet34
from torchvision import transforms

from config import Params
from layers import YoloConvBlockNaive, YoloDetectionBlockNaive, FeatureFusionNaive, ConvNormRelu
from DetectionNetAbstract import DetectionNet

class DetectionNetNaive(DetectionNet):
	def __init__(self, backbone_pretrained, backbone_freeze, device='cpu'):
		super(DetectionNetNaive, self).__init__(backbone_pretrained, backbone_freeze, device)
		""" Large Detect """
		self.large_detect_cnr = ConvNormRelu(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0)
		self.large_detect_conv_block = YoloConvBlockNaive(in_channels=512, filters=512)
		self.large_detect_detect_block = YoloDetectionBlockNaive(in_channels=1024)

		""" Mid Detect """
		self.mid_detect_cnr = ConvNormRelu(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0)
		self.mid_detect_fusion = FeatureFusionNaive(low_res_channels=512)
		self.mid_detect_conv_block = YoloConvBlockNaive(in_channels=512, filters=256)
		self.mid_detect_detect_block = YoloDetectionBlockNaive(in_channels=512)

		""" Small Detect """
		self.small_detect_cnr = ConvNormRelu(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0)
		self.small_detect_fusion = FeatureFusionNaive(low_res_channels=256)
		self.small_detect_conv_block = YoloConvBlockNaive(in_channels=256, filters=128)
		self.small_detect_detect_block = YoloDetectionBlockNaive(in_channels=256)



		self.to(device)
		self.device = device

	def forward(self, x):

		# x = torch.randn(1, 3, 448, 448)
		x = self.transforms(x)
		r1, r2, r3 = self.backbone(x)
		# print(r1.shape, r2.shape, r3.shape)

		# r1 = torch.randn(1, 64, 112, 112)
		# r2 = torch.randn(1, 128, 56, 56)
		# r3 = torch.randn(1, 256, 28, 28)

		""" Large Detection """
		r3 = self.large_detect_cnr(r3) # [n,512,14,14]
		route, out = self.large_detect_conv_block(r3) #[n,512,14,14], [n,1024,14,14]
		detect_large = self.large_detect_detect_block(out) #[n,5,14,14]

		""" Mid Detection """
		r2 = self.mid_detect_cnr(r2) # [n,256,28,28]
		out = self.mid_detect_fusion(route, r2) # [n,512,28,28]
		route, out = self.mid_detect_conv_block(out) #[n,256,28,28], [n,512,28,28]
		detect_mid = self.mid_detect_detect_block(out) #[n,5,28,28]

		""" Small Detection """
		r1 = self.small_detect_cnr(r1) # [n,128,56,56]
		out = self.small_detect_fusion(route, r1) # [n,256,56,56]
		route, out = self.small_detect_conv_block(out) # [n,128,56,56], [n,256,56,56]
		detect_small = self.small_detect_detect_block(out) # [n,5,56,56]

		# print(detect_large.shape, detect_mid.shape, detect_small.shape)

		return detect_large, detect_mid, detect_small