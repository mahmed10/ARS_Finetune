#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import os
from PIL import Image
import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import gc
import sys
import warnings

from config import cfg
from utils.func import set_deterministic, adjust_learning_rate, loss_calc
from utils.data_organize import prepare_dataset, prepare_datalist, prepare_dataloader
from model.deeplabv2 import get_deeplab_v2


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")



set_deterministic(seed=42)

start_fineTune = False
def button_input_callback(data):
	# Callback function for handling incoming messages from /button_input
	message = data.data
	global start_fineTune
	if message == "5":
		start_fineTune = True

def finetune(cfg):
	# Create the directory
	# The 'exist_ok' parameter is set to True, which means the function won't raise an error if the directory already exists.
	os.makedirs(cfg.DATA_DIR + '/Old/Images', exist_ok=True)
	os.makedirs(cfg.DATA_DIR + '/New/Images', exist_ok=True)
	os.makedirs(cfg.DATA_DIR + '/New/LabelMasks', exist_ok=True)

	 
	with open(cfg.OUTPUR_DIR + '/meta.txt', 'r') as file:
		lines = file.readlines()
	# # Sort the lines based on the entropy value (which is the first value on each line)
	sorted_lines = sorted(lines, key=lambda line: float(line.split(",")[0]), reverse = True)

	for line in sorted_lines:
		prepare_dataset(line, cfg)
	prepare_datalist(cfg)

	trainloader, targetloader = prepare_dataloader(cfg)

	input_size_source = cfg.INPUT_SIZE_SOURCE
	input_size_target = cfg.INPUT_SIZE
	device = cfg.GPU_ID
	num_classes = cfg.NUM_CLASSES

	model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.MULTI_LEVEL[0])
	checkpoint = torch.load(cfg.checkpoint)
	model.load_state_dict(checkpoint)
	model.train()
	model.to(device)
	cudnn.benchmark = True
	cudnn.enabled = True

	model_clone=copy.deepcopy(model)
	model_clone.to(device)
	for param in model_clone.parameters():
		param.requires_grad = False
	model_clone.eval()

	# OPTIMIZERS
	# segnet's optimizer
	optimizer = optim.SGD(model.optim_parameters(cfg.LEARNING_RATE),
		lr=cfg.LEARNING_RATE,
		momentum=cfg.MOMENTUM,
		weight_decay=cfg.WEIGHT_DECAY)

	# interpolate output segmaps
	interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

	trainloader_iter = enumerate(trainloader)
	targetloader_iter = enumerate(targetloader)

	for i_iter in tqdm(range(cfg.EARLY_STOP + 1)):
		# torch.cuda.empty_cache()
		if (i_iter < 95 and i_iter>5 and i_iter%10==0):
			key_pub.publish("FineTune "+repr(i_iter)+"% Done")
		optimizer.zero_grad()
		adjust_learning_rate(optimizer, i_iter, cfg)

		# train on source
		_, batch = trainloader_iter.__next__()
		images_source, image_size, _ = batch
		if image_size[0,1].item() == 720:
			interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
		elif image_size[0,1].item() == 480:
			interp = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
		# with torch.no_grad():
			# labels_source = model_clone(images_source.cuda(device))[1]
			# labels_source = interp(labels_source)
			# labels_source = torch.argmax(labels_source, axis=1)
		pred_src_aux, pred_src_main = model(images_source.cuda(device))
		labels_source = interp(pred_src_aux)
		labels_source = torch.argmax(labels_source, axis=1)
		del images_source
		if cfg.MULTI_LEVEL:
			pred_src_aux = interp(pred_src_aux)
			loss_seg_src_aux = loss_calc(pred_src_aux, labels_source, device)
			del pred_src_aux
		else:
			loss_seg_src_aux = 0
		pred_src_main = interp(pred_src_main)
		loss_seg_src_main = loss_calc(pred_src_main, labels_source, device)
		del pred_src_main
		del labels_source
		# del loss_seg_src_main
		# del loss_seg_src_aux
		# gc.collect()
		# torch.cuda.empty_cache()
		# 



		# train on target
		_, batch = targetloader_iter.__next__()
		images_target, labels_target, _, _ = batch
		pred_trg_aux, pred_trg_main = model(images_target.cuda(device))
		if cfg.MULTI_LEVEL:
			pred_trg_aux = interp_target(pred_trg_aux)
			loss_seg_trg_aux = loss_calc(pred_trg_aux, labels_target, device)
		else:
			loss_seg_src_aux = 0
		pred_trg_main = interp_target(pred_trg_main)
		loss_seg_trg_main = loss_calc(pred_trg_main, labels_target, device)

		loss = (cfg.LAMBDA_SEG_MAIN * loss_seg_src_main
			+ cfg.LAMBDA_SEG_AUX * loss_seg_src_aux
			+ cfg.LAMBDA_SEG_MAIN * loss_seg_trg_main
			+ cfg.LAMBDA_SEG_AUX * loss_seg_trg_aux)
		# print(loss_seg_src_main)
		# logger.info(f'iter = {str(i_iter).zfill(6)} '
		#             + f'loss_seg_src_main = {loss_seg_src_main: .5f} '
		#             + f'loss_seg_src_aux = {loss_seg_src_aux: .5f} '
		#             + f'loss_seg_trg_main = {loss_seg_trg_main: .5f} '
		#             + f'loss_seg_trg_aux = {loss_seg_trg_aux: .5f} ')
		loss.backward()
		optimizer.step()
	sys.stdout.flush()
	key_pub.publish("FineTune 100% Done")
	torch.save(model.state_dict(), cfg.checkpoint) 
	global start_fineTune
	start_fineTune = False


rospy.init_node('fine_tune', anonymous=True)
button_input_sub = rospy.Subscriber('/button_input', String, button_input_callback)
key_pub = rospy.Publisher('button_input', String, queue_size=10)


while not rospy.is_shutdown():
	if not start_fineTune:
		continue
	finetune(cfg)