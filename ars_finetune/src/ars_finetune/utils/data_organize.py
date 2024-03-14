import os
from PIL import Image
import numpy as np
from glob import glob
import random
from torch.utils import data
from torch.utils.data import ConcatDataset

from dataset.umbc import UmbcDataSet
from dataset.gta5 import GTA5DataSet

def prepare_dataset(line, cfg):
	file_name = line.split(',')[1]
	if os.path.exists(cfg.OUTPUR_DIR + '/images/' + file_name.replace('.png', '_watershed_mask.png')):
		image = Image.open(cfg.OUTPUR_DIR + '/images/' + file_name)
		image = np.asarray(image, np.float32)

		mask = Image.open(cfg.OUTPUR_DIR + '/images/' + file_name.replace('.png', '_watershed_mask.png')).convert('L')
		mask = np.asarray(mask, np.float32)
		mask[mask == 255] = 0

		y = int(line.split(',')[2])
		x = int(line.split(',')[3])
		
		new_array = np.zeros((480, 640, 3), dtype=np.float32)
		# Place 'a' within new_array at the specified coordinates
		new_array[y:y+240, x:x+320] = image

		# Convert the numpy array back to a PIL Image
		img = Image.fromarray(np.uint8(new_array))

		# Save the PIL Image to a file
		img.save(cfg.DATA_DIR + '/New/Images/' + file_name)

		new_array = np.zeros((480, 640), dtype=np.float32)
		# Place 'a' within new_array at the specified coordinates
		new_array[y:y+240, x:x+320] = mask

		# Convert the numpy array back to a PIL Image
		img = Image.fromarray(np.uint8(new_array))

		# Save the PIL Image to a file
		img.save(cfg.DATA_DIR + '/New/LabelMasks/' + file_name.replace('.png', '_label_mask.png'))

def prepare_list(file_path, data_dir):
	random.shuffle(data_dir)
	with open(file_path, 'w') as file:
		for i, item in enumerate(data_dir):
			file.write(str(item.split('Images/')[1]))
			if i < len(data_dir) - 1:
				file.write('\n')

def prepare_datalist(cfg):
	data_dir = sorted(glob(cfg.DATA_DIR + '/New/Images/*.png'))
	prepare_list(cfg.ROOT_DIR+'/dataset/umbc_list_new.txt', data_dir)

	data_dir = sorted(glob(cfg.DATA_DIR + '/Old/Images/*.png'))
	prepare_list(cfg.ROOT_DIR+'/dataset/umbc_list_old.txt', data_dir)

	data_dir = sorted(glob(cfg.DATA_DIR + '/GTA5/Images/*.png'))
	prepare_list(cfg.ROOT_DIR+'/dataset/gta5_list.txt', data_dir)

def prepare_dataloader(cfg):
	source_dataset = []
	source_dataset.append(GTA5DataSet(root=cfg.DATA_DIR,
		list_path=cfg.ROOT_DIR+'/dataset/gta5_list.txt',
		set='all',
		max_iters=cfg.MAX_ITERS * cfg.BATCH_SIZE,
		crop_size=cfg.INPUT_SIZE_SOURCE,
		mean=cfg.IMG_MEAN))
	source_dataset.append(UmbcDataSet(root=cfg.DATA_DIR,
		list_path=str(cfg.ROOT_DIR)+'/dataset/umbc_list_old.txt',
		set='Old',
		info_path=cfg.INFO,
		max_iters=cfg.MAX_ITERS * cfg.BATCH_SIZE,
		crop_size=cfg.INPUT_SIZE,
		mean=cfg.IMG_MEAN))

	trainloader = data.DataLoader(ConcatDataset(source_dataset),
		batch_size=cfg.BATCH_SIZE,
		num_workers=cfg.NUM_WORKERS,
		shuffle=True,
		pin_memory=True,
		worker_init_fn=None)

	target_dataset = UmbcDataSet(root=cfg.DATA_DIR,
		list_path=str(cfg.ROOT_DIR)+'/dataset/umbc_list_new.txt',
		set='New',
		info_path=cfg.INFO,
		max_iters=cfg.MAX_ITERS * cfg.BATCH_SIZE,
		crop_size=cfg.INPUT_SIZE,
		mean=cfg.IMG_MEAN)

	targetloader = data.DataLoader(target_dataset,
		batch_size=cfg.BATCH_SIZE,
		num_workers=cfg.NUM_WORKERS,
		shuffle=True,
		pin_memory=True,
		worker_init_fn=None)

	return trainloader, targetloader