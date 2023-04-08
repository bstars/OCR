import torch
from torch import nn
import numpy as np
import abc

from torchvision.models import resnet34
from torchvision import transforms

from config import Params


class SplitResnet(nn.Module):
	def __init__(self, pretrained=True, freeze=True, device='cpu'):
		super(SplitResnet, self).__init__()
		# backbone = vgg16(pretrained).to(device)
		backbone = resnet34(pretrained=pretrained).to(device)
		self.net1 = nn.Sequential(
			backbone.conv1.to(device),
			backbone.bn1.to(device),
			backbone.relu.to(device),
			backbone.maxpool.to(device),
			backbone.layer1.to(device)
		).to(device)
		self.net2 = nn.Sequential( backbone.layer2.to(device) ).to(device)
		self.net3 = nn.Sequential( backbone.layer3.to(device) ).to(device)
		self.to(device)

		if freeze:
			for p in self.parameters():
				p.requires_grad = False

	def forward(self, x):

		route1 = self.net1(x)
		route2 = self.net2(route1)
		out = self.net3(route2)

		return route1, route2, out

class DetectionNet(nn.Module):
	def __init__(self, backbone_pretrained, backbone_freeze, device='cpu'):
		super(DetectionNet, self).__init__()
		self.img_size = Params.IMG_SIZE
		self.object_scale = Params.OBJECT_SCALE
		self.noobject_scale = Params.NOOBJECT_SCALE
		self.coord_scale = Params.COORD_SCALE

		self.register_buffer('anchors', torch.from_numpy(Params.ANCHORS).to(device))

		self.large_obj_scale = Params.LARGE_OBJ_SCALE
		self.mid_obj_scale = Params.MID_OBJ_SCALE
		self.small_obj_scale = Params.SMALL_OBJ_SCALE

		self.backbone = SplitResnet(pretrained=backbone_pretrained, freeze=backbone_freeze, device=device)

		self.transforms = transforms.Compose([
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])

	@abc.abstractmethod
	def forward(self, x):
		pass

	def get_offset(self, cell_size):
		x = torch.arange(0, cell_size)
		y = torch.arange(0, cell_size)
		xx, yy = torch.meshgrid(x, y)
		offset = torch.stack([yy, xx])
		return offset.to(self.device)

	def compute_ious(self, boxes1, boxes2):
		"""
		:param boxes1: [n,4,cell_size,cell_size] in absolute pixel
		:param boxes2: [n,4,cell_size,cell_size] in absolute pixel
		:return:
		"""
		boxes1_tran = torch.stack([
			boxes1[:,0,:,:] - boxes1[:,2,:,:]/2, # x of upper-left in absolute pixel
			boxes1[:,1,:,:] - boxes1[:,3,:,:]/2, # y of upper-left in absolute pixel
			boxes1[:,0,:,:] + boxes1[:,2,:,:]/2,
			boxes1[:,1,:,:] + boxes1[:,3,:,:]/2
		], dim=1)

		boxes2_tran = torch.stack([
			boxes2[:,0,:,:] - boxes2[:,2,:,:]/2,
			boxes2[:,1,:,:] - boxes2[:,3,:,:]/2,
			boxes2[:,0,:,:] + boxes2[:,2,:,:]/2,
			boxes2[:,1,:,:] + boxes2[:,3,:,:]/2
		], dim=1)

		upper_left = torch.maximum(boxes1_tran[:,:2,:,:],  boxes2_tran[:,:2,:,:])
		lower_right = torch.minimum(boxes1_tran[:,2:,:,:], boxes2_tran[:,2:,:,:])

		inter_wh = torch.maximum(lower_right - upper_left, torch.zeros_like(upper_left).to(self.device))
		inter_area = inter_wh[:,0,:,:] * inter_wh[:,1,:,:]

		area1 = boxes1[:,2,:,:] * boxes1[:,3,:,:]
		area2 = boxes2[:,2,:,:] * boxes2[:,3,:,:]
		union_area = torch.maximum(area1 + area2 - inter_area, torch.ones_like(area2).to(self.device) * 1e-10)
		return torch.clamp(inter_area / union_area, 0., 1.)

	def compute_loss_one_output(self, output, label, anchor):
		"""
		:param output: [n,5,cell_size, cell_size]
		:param label: [n,5,cell_size,cell_size]
		:param anchor: [2,]
		:return:
		"""
		cell_size = output.shape[-1]
		px_per_cell = float(self.img_size) / cell_size
		offset = self.get_offset(cell_size)

		anchor = torch.stack([anchor for _ in range(cell_size)], dim=-1)
		anchor = torch.stack([anchor for _ in range(cell_size)], dim=-1)

		# [n,1,cell_size,cell_size], # [n,2,cell_size,cell_size], # [n,2,cell_size,cell_size]
		confidence, xy, wh = torch.split(output, [1,2,2], dim=1)
		true_confidence, true_xy, true_wh = torch.split(label, [1,2,2], dim=1) #


		""" Detection Loss """
		predict_absolute_pixels = torch.stack([
			(output[:,1,:,:] + offset[0,:,:]) * px_per_cell, # absolute x of box center
			(output[:,2,:,:] + offset[1,:,:]) * px_per_cell, # absolute y of box center
			output[:,3,:,:] * anchor[0,:,:],
			output[:,4,:,:] * anchor[1,:,:]
		], dim=1) # [n, 4, cell_size, cell_size]

		ious = self.compute_ious(predict_absolute_pixels, label[:,1:,:,:]) # [n,cell_size, cell_size]

		object_mask = true_confidence
		noobject_mask = 1 - true_confidence

		object_loss = ((confidence[:,0,:,:] - ious) * object_mask[:,0,:,:])**2
		object_loss = torch.mean( torch.sum(object_loss, dim=[1,2]) )

		noobject_loss = (confidence[:,0,:,:] * noobject_mask)**2
		noobject_loss = torch.mean( torch.sum(noobject_loss, dim=[1,2]) )

		detection_loss = self.object_scale * object_loss + self.noobject_scale * noobject_loss


		""" bounding box loss """
		true_xy_tran = true_xy / px_per_cell - offset
		true_wh_tran = true_wh / anchor

		xy_loss = torch.sum( (xy - true_xy_tran)**2, dim=1 )
		xy_loss = torch.mean( torch.sum(xy_loss * object_mask[:,0,:,:], dim=[1,2]) )

		wh_loss = torch.sum( (wh - true_wh_tran)**2, dim=1 )
		wh_loss = torch.mean( torch.sum(wh_loss * object_mask[:,0,:,:], dim=[1,2]) )

		box_loss = self.coord_scale * (xy_loss + wh_loss)

		return box_loss  + detection_loss

	def compute_loss(self, output_large, output_mid, output_small,
	                    label_large, label_mid, label_small):
		"""
		:param output_large: [n, 5, 14, 14]
		:param output_mid:  [n, 5, 28, 28]
		:param output_small: [n, 5, 56, 56]
		:param label_large: [n, 5, 14, 14]
		:param label_mid: [n, 5, 28, 28]
		:param label_small: [n, 5, 56, 56]
		:return:
		"""
		loss_large = self.compute_loss_one_output(output_large, label_large, self.anchors[0,:])
		loss_mid = self.compute_loss_one_output(output_mid, label_mid, self.anchors[1,:])
		loss_small = self.compute_loss_one_output(output_small, label_small, self.anchors[2,:])
		return loss_large * self.large_obj_scale \
		       + loss_mid * self.mid_obj_scale\
		       + loss_small * self.small_obj_scale

	def interpret_one_output(self, output, anchor):
		"""
		Convert detection output to absolute pixel
		:param output:
		:return:
		"""
		cell_size = output.shape[-1]
		px_per_cell = float(self.img_size) / cell_size
		confidence, xy, wh = torch.split(output, [1,2,2], dim=1)
		offset = self.get_offset(cell_size)

		anchor = torch.stack([anchor for _ in range(cell_size)], dim=-1)
		anchor = torch.stack([anchor for _ in range(cell_size)], dim=-1)

		xy = (xy + offset) * px_per_cell
		wh = wh * anchor

		return torch.cat([confidence, xy, wh], dim=1)

	def interpret_outputs(self, output_large, output_mid, output_small):
		prediction_large = self.interpret_one_output(output_large, self.anchors[0,:])
		prediction_mid = self.interpret_one_output(output_mid, self.anchors[1,:])
		prediction_small = self.interpret_one_output(output_small, self.anchors[2,:])
		return prediction_large, prediction_mid, prediction_small

# class DetectionNetNaive(DetectionNet):
# 	def __init__(self, backbone_pretrained, backbone_freeze, device='cpu'):
# 		super(DetectionNetNaive, self).__init__(backbone_pretrained, backbone_freeze, device)
# 		""" Large Detect """
# 		self.large_detect_cnr = ConvNormRelu(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0)
# 		self.large_detect_conv_block = YoloConvBlockNaive(in_channels=512, filters=512)
# 		self.large_detect_detect_block = YoloDetectionBlockNaive(in_channels=1024)
#
# 		""" Mid Detect """
# 		self.mid_detect_cnr = ConvNormRelu(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0)
# 		self.mid_detect_fusion = FeatureFusionNaive(low_res_channels=512)
# 		self.mid_detect_conv_block = YoloConvBlockNaive(in_channels=512, filters=256)
# 		self.mid_detect_detect_block = YoloDetectionBlockNaive(in_channels=512)
#
# 		""" Small Detect """
# 		self.small_detect_cnr = ConvNormRelu(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0)
# 		self.small_detect_fusion = FeatureFusionNaive(low_res_channels=256)
# 		self.small_detect_conv_block = YoloConvBlockNaive(in_channels=256, filters=128)
# 		self.small_detect_detect_block = YoloDetectionBlockNaive(in_channels=256)
#
#
#
# 		self.to(device)
# 		self.device = device
#
# 	def forward(self, x):
#
# 		# x = torch.randn(1, 3, 448, 448)
# 		x = self.transforms(x)
# 		r1, r2, r3 = self.backbone(x)
# 		# print(r1.shape, r2.shape, r3.shape)
#
# 		# r1 = torch.randn(1, 64, 112, 112)
# 		# r2 = torch.randn(1, 128, 56, 56)
# 		# r3 = torch.randn(1, 256, 28, 28)
#
# 		""" Large Detection """
# 		r3 = self.large_detect_cnr(r3) # [n,512,14,14]
# 		route, out = self.large_detect_conv_block(r3) #[n,512,14,14], [n,1024,14,14]
# 		detect_large = self.large_detect_detect_block(out) #[n,5,14,14]
#
# 		""" Mid Detection """
# 		r2 = self.mid_detect_cnr(r2) # [n,256,28,28]
# 		out = self.mid_detect_fusion(route, r2) # [n,512,28,28]
# 		route, out = self.mid_detect_conv_block(out) #[n,256,28,28], [n,512,28,28]
# 		detect_mid = self.mid_detect_detect_block(out) #[n,5,28,28]
#
# 		""" Small Detection """
# 		r1 = self.small_detect_cnr(r1) # [n,128,56,56]
# 		out = self.small_detect_fusion(route, r1) # [n,256,56,56]
# 		route, out = self.small_detect_conv_block(out) # [n,128,56,56], [n,256,56,56]
# 		detect_small = self.small_detect_detect_block(out) # [n,5,56,56]
#
# 		print(detect_large.shape, detect_mid.shape, detect_small.shape)
#
# 		return detect_large, detect_mid, detect_small