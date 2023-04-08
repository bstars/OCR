import torch
import numpy as np

class Params:

	""" Data Parameter """
	IMG_SIZE = 448

	ANCHORS = np.array([ # w,h
		[280, 320],
		[120, 190],
		[70, 100]
	])

	# These anchor boxes are from YOLO V3 but are pretty compatible with the face sizes in the dataset

	# Grid size for different-scale detection
	SMALL_SIZE = 14
	MEDIUM_SIZE = 28
	LARGE_SIZE = 56

	FDDB_PATH = './FDDB'
	# FDDB_PATH = '/kaggle/input/fddbdataset'
	UNLABELED_DATA_PATH = './memotion_dataset_7k/images'

	CHECKPOINTS_SAVING_PATH = './'


	""" Loss Parameters """
	OBJECT_SCALE = 50.
	NOOBJECT_SCALE = 0.1 # 0.5
	COORD_SCALE = 30.

	LARGE_OBJ_SCALE = 1.05
	MID_OBJ_SCALE = 1.
	SMALL_OBJ_SCALE = 1.05

	UNSUPER_SCALE = 1e-3


	# training parameter
	BATCH_SIZE = 32
	OPTIM = torch.optim.Adam
	LEARNING_RATE = 1e-4
	SEMI_LEARNING_RATE = 1e-4

	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

	""" Inference Parameters """
	NAIVE_CKPT_PATH = 'ckpts/naive_120.pth'
	MULTIHEAD_CKPT_PATH = 'ckpts/multihead_120.pth'
	MULTIHEAD_CKPT_SYM_PATH = 'ckpts/70_sym.pth'
	CONFIDENCE_THRESHOLD = 0.55
	IOU_THRESHOLD_NONMAX_SUPRESSION = 0.4
	IOU_THRESHOLD_MAP = 0.65
