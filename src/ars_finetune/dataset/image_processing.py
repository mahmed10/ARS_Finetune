import io
from PIL import Image
import numpy as np
import torch

def image_processing(image, mean):
	image = image[:, :, ::-1]  # change to BGR
	image -= mean
	image = image.transpose((2, 0, 1))
	return torch.from_numpy(image.copy()).unsqueeze(0) 

def image_load(data, size):
	image_stream = io.BytesIO(data.data)
	img = Image.open(image_stream)
	img = img.convert('RGB')
	img = img.resize(size, Image.BICUBIC)
	return np.asarray(img, np.float32)