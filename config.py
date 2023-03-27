import torch




""" https://www.kaggle.com/datasets/eabdul/textimageocr """
class Config:
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RecognitionConfig:
	TOKENIZER_PATH = '../datasets/tokenizer.txt'
	H = 130
	PX_PER_WINDOW = 30