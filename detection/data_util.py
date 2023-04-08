import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Params

class Unlabeled(Dataset):
	def __init__(self):
		self.data_path = Params.UNLABELED_DATA_PATH
		self.img_size = Params.IMG_SIZE
		self.filenames = []
		for path, directories, files in os.walk(self.data_path):
			for f in files:
				if f == '.DS_Store':
					continue
				self.filenames.append(
					os.path.join(path, f)
				)
		self.transforms = transforms.Compose([
			transforms.ToTensor(),
		])

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		# plt.imread(self.filenames[idx])
		try:
			img = cv2.imread(self.filenames[idx])
			img = cv2.resize(img, (self.img_size, self.img_size))
			img = img[:, :, [2, 1, 0]]
			return self.transforms(img.copy())
		except cv2.error:
			print(self.filenames[idx])

class FDDB(Dataset):
	def __init__(self, test=False):
		self.small_size = Params.SMALL_SIZE
		self.mid_size = Params.MEDIUM_SIZE
		self.large_size = Params.LARGE_SIZE
		self.img_size = Params.IMG_SIZE
		self.augments = ['IDENTITY', 'BLUR', 'COLOR', 'AVERAGE', 'GAUSSIAN', "MEDIAN"]

		self.test = test
		if test:
			self.folds = ['10']
		else:
			self.folds = ['01', '02', '03', '04', '05', '06', '07', '08', '09']

		self.path = Params.FDDB_PATH
		self.anchors = Params.ANCHORS

		self.transforms = transforms.Compose([transforms.ToTensor()])

		idxes = []
		for fold in self.folds:
			idx_filename = os.path.join(self.path, 'FDDB-folds', 'FDDB-fold-' + fold + '.txt')
			idx_file = open(idx_filename)

			for line in idx_file:
				idxes.append(line.strip())
		self.index = idxes

		self.transforms = transforms.Compose([
			transforms.ToTensor(),
		])

	def get_image_with_boxes(self, idx):
		img_path = os.path.join(self.path, 'originalPics', self.index[idx] + '.jpg')
		label_path = os.path.join(self.path, 'labels', self.index[idx] + '.txt')

		img = cv2.imread(img_path)
		original_h, original_w, _ = img.shape
		img = cv2.resize(img, (self.img_size, self.img_size))
		img = img[:, :, [2, 1, 0]]

		label_file = open(label_path)
		boxes = []
		for line in label_file:
			coord = np.array(line.strip().split()).astype(float)
			# print(idx, label_path, coord)
			h, w, x, y = coord[0] * 2, coord[1] * 2, coord[3], coord[4]
			x = x / original_w * self.img_size
			y = y / original_h * self.img_size
			w = w / original_w * self.img_size
			h = h / original_h * self.img_size
			boxes.append([x, y, w, h])

		# boxes are [-1, [centerx, centery, w, h]]
		return img, np.array(boxes)

	def __len__(self):
		return len(self.index)

	def __getitem__(self, idx):
		img, boxes = self.get_image_with_boxes(idx)
		if not self.test:
			img, boxes = self.augment_data(img.copy(), boxes)
		label_large_obj = np.zeros(shape=[5, self.small_size, self.small_size], dtype=np.float32)
		label_mid_obj = np.zeros(shape=[5, self.mid_size, self.mid_size], dtype=np.float32)
		label_small_obj = np.zeros(shape=[5, self.large_size, self.large_size], dtype=np.float32)

		for box in boxes:
			ious = iou_same_center([box[2], box[3]], self.anchors)
			best_iou_idx = np.argmax(ious)

			if best_iou_idx == 0:   # large object, small grid
				px_per_cell = self.img_size / self.small_size
				x_idx = int( (box[0]-1) / px_per_cell )
				y_idx = int( (box[1]-1) / px_per_cell )
				label_large_obj[0,y_idx, x_idx] = 1
				label_large_obj[1:,y_idx, x_idx] = box

			elif best_iou_idx == 1:   # mid object, mid grid
				px_per_cell = self.img_size / self.mid_size
				x_idx = int( (box[0]-1) / px_per_cell )
				y_idx = int( (box[1]-1) / px_per_cell )
				label_mid_obj[0,y_idx, x_idx] = 1
				label_mid_obj[1:,y_idx, x_idx] = box

			elif best_iou_idx == 2:   # small object, large grid
				px_per_cell = self.img_size / self.large_size
				x_idx = int( (box[0]-1) / px_per_cell )
				y_idx = int( (box[1]-1) / px_per_cell )
				label_small_obj[0,y_idx, x_idx] = 1
				label_small_obj[1:,y_idx, x_idx] = box

		return self.transforms(img.copy()), label_large_obj, label_mid_obj, label_small_obj

	def augment_data(self, img, boxes):

		u = np.random.uniform(0,1,2)
		if u[0] >= 0.5: return img, boxes
		if u[1] >= 0.5: img = img[:, ::-1, :]; boxes[:, 0, ] = self.img_size - boxes[:, 0]

		u = np.random.randint(0, 5)
		if u == 0: cv2.GaussianBlur(img, (5,5), 3, 3)
		if u == 1: img = img[:,:,[2,1,0]]
		if u == 2: img = cv2.medianBlur(img, 5)
		if u == 3: img = cv2.blur(img, (5, 5))


		return img, boxes

	# def augment_data(self, img, boxes):
	# 	# u = np.random.uniform(0, 1, 2)
	# 	#
	# 	img = img[:, ::-1, :]; boxes[:, 0, ] = self.img_size - boxes[:, 0]
	#
	# 	img = cv2.medianBlur(img, 5)
	# 	img = img[:, :, [2, 1, 0]]
	# 	return img, boxes

def iou_same_center(box1, boxes):
	"""
	Select from boxes the one which has the highest iou with box1
	Assuming all boxes have the same center
	:param box1: of shape [2,], [w,h]
	:param boxes: of shape [n,2], [n,[w,h]]
	:return:
	"""
	boxes1 = np.stack([box1 for i in range(len(boxes))], axis=0)
	w1 = boxes1[:,0]
	h1 = boxes1[:,1]
	w2 = boxes[:,0]
	h2 = boxes[:,1]

	w_inter = np.minimum(w1, w2)
	h_inter = np.minimum(h1, h2)

	intersection = w_inter * h_inter
	union = h1 * w1 + h2 * w2 - intersection
	return intersection / union

def plot_img_with_boxes(img, boxes, anchors=None):
	"""
	:param img: np.array, [448, 448, 3]
	:param boxes:
	"""

	fig, ax = plt.subplots(1)
	ax.imshow(img)
	for box in boxes:
		xcenter, ycenter, w, h = box
		x1, y1 = xcenter-w/2, ycenter-h/2
		x2, y2 = xcenter+w/2, ycenter+h/2

		rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=3, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

		if anchors is not None:
			ious = iou_same_center([w,h], anchors)
			idx = np.argmax(ious)
			print(idx)
			w_anchor, h_anchor = anchors[idx, 0], anchors[idx, 1]
			rect = patches.Rectangle(
				(xcenter - w_anchor/2, ycenter - h_anchor/2), w_anchor, h_anchor, linewidth=1, edgecolor='b', facecolor='none'
			)
			ax.add_patch(rect)

	plt.show()

def plot_img_with_label(img, label_large, label_mid, label_small):
	boxes = []
	for label in [label_large, label_mid, label_small]:
		confidence = label[0,:,:]
		Is, Js = np.where(confidence >= 0.5)
		for i in range(len(Is)):
			box = label[1:, Is[i], Js[i]]
			boxes.append(box)
	plot_img_with_boxes(img, boxes)

if __name__ == '__main__':
	fddb = FDDB(False)
	for i in range(40, 600):
		# img, boxes = fddb.get_image_with_boxes(i)
		# plot_img_with_boxes(img, boxes, anchors=Params.ANCHORS)


		# img, label_large, label_mid, label_small = fddb[i]
		# plot_img_with_label(img.transpose(), label_large, label_mid, label_small)

		img, boxes = fddb.get_image_with_boxes(i)
		plot_img_with_boxes(img, boxes)

		img, boxes = fddb.augment_data(img, boxes)
		plot_img_with_boxes(img, boxes)
		# plot_img_with_boxes(np.ones_like(img).astype(int) * 255, boxes)


