#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage, Image as ROSImage
from jackal_msgs.msg import Status
from std_msgs.msg import String
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from cv_bridge import CvBridge
import cv2
import time
from datetime import datetime
import warnings
import os
import psutil

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

from config import cfg
from model.deeplabv2 import get_deeplab_v2
from utils import project_root
from utils.func import prob_2_entropy, find_rare_class
from utils.visualization import colorize_mask
from dataset.image_processing import image_processing, image_load

bridge = CvBridge()
current_value = 1.0
voltage_value = 2.9
start_segmentation = False

def status_callback(msg):
	global current_value
	global voltage_value
	current_value = float(str(msg.total_current))
	voltage_value = float(str(msg.measured_battery))
def button_input_callback(data):
	# Callback function for handling incoming messages from /button_input
	message = data.data
	global start_segmentation
	if message == "1":
		start_segmentation = True
		os.makedirs(cfg.OUTPUR_DIR + '/images', exist_ok=True)
	if message == "2":
		start_segmentation = False
	if message == "3":
		start_segmentation = False


def semantic_seg(data):
	# rospy.Rate(30.0)
	process = psutil.Process(os.getpid())
	start_time = time.time()
	image = image_load(data, cfg.INPUT_SIZE)
	image = image_processing(image, cfg.IMG_MEAN)

	with torch.no_grad():
		pred_main = model(image.cuda(device))[1]
		pred_main = interp(pred_main)

		output = pred_main.cpu().data[0].numpy()
		assert output is not None, 'Output is None'
		output = output.transpose(1, 2, 0)
		output = np.argmax(output, axis=2)

		pred_entropy = prob_2_entropy(F.softmax(pred_main))
		normalizor = len(find_rare_class(pred_main)) / 11.0 + 0.5
		entropy = pred_entropy.mean().item() * normalizor


		pred_entropy = pred_entropy.mean(dim=1)
		mask = pred_entropy.unsqueeze(1)
		mask = F.avg_pool2d(mask, (241, 321), stride=(1, 1))
	max_idx = torch.argmax(mask)
	_, _, y, x = max_idx // mask.shape[1] // mask.shape[2] // mask.shape[3], \
				(max_idx // mask.shape[2] // mask.shape[3]) % mask.shape[1], \
				(max_idx // mask.shape[3]) % mask.shape[2], \
				max_idx % mask.shape[3]

	mask = colorize_mask(output)
	mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

	# Convert the numpy RGB image to an ROS Image message
	ros_image_msg = bridge.cv2_to_compressed_imgmsg(mask, dst_format="jpeg")
	
	# Publish the ROS Image message
	mask_pub.publish(ros_image_msg)

	image = np.fromstring(data.data, np.uint8)
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	image = image[y: y + 240, x:x + 320, :]
	# Save the original RGB image
	timestamp = datetime.now().strftime('%Y%m%d_%H_%M_%S_') + str(int(datetime.now().microsecond / 1000)).zfill(3)
	image_path = f'{cfg.OUTPUR_DIR}/images/{timestamp}.png'
	cv2.imwrite(image_path, image)
	rospy.loginfo("Saved file: %s", image_path)

	# Save entropy, image path, y, x into a text file
	file_path = os.path.join(cfg.OUTPUR_DIR, 'info.txt')
	with open(file_path, 'a') as file:
		file.write(f"{entropy},{image_path.split('/')[-1]},{y},{x}\n")
	end_time = time.time()
	string = f"{timestamp}, {round((end_time - start_time)*1000,2)}, "
	ram = process.memory_info().rss / 1024 / 1024
	global current_value
	global voltage_value
	string += f"{round(ram,2)}, {round(current_value,2)}, {round(voltage_value,2)}"
	# Save powerfile
	file_path = os.path.join(cfg.OUTPUR_DIR, 'powercosump.txt')
	with open(file_path, 'a') as file:
		file.write(f"{string}\n")

device = cfg.GPU_ID

model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.MULTI_LEVEL[0])
checkpoint = torch.load(cfg.checkpoint)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

interp = nn.Upsample(size=(cfg.OUTPUT_SIZE[1], cfg.OUTPUT_SIZE[0]), mode='bilinear', align_corners=True)



rospy.loginfo("Started segmentation")
image_data = None
def image_call(data):
	global image_data
	image_data = data

rospy.init_node('semantic_segmentation', anonymous=True)
rospy.Subscriber("/status", Status, status_callback, queue_size=1)
mask_pub = rospy.Publisher('/segmentation/mask', CompressedImage, queue_size=1)
rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, image_call, queue_size=1)
button_input_sub = rospy.Subscriber('/button_input', String, button_input_callback)
# rospy.spin()
while not rospy.is_shutdown():
	rospy.Rate(30.0).sleep()
	# print(start_segmentation)
	if image_data is None:
		continue
	if not start_segmentation:
		continue
	semantic_seg(image_data)

# running = False
# while not rospy.is_shutdown():
# 	if rospy.get_param('start_segmentation', True):
# 		running = True
# 		print('okay got it')
# 		rospy.set_param('start_segmentation', False)  # Reset the param

# 	if running:
# 		rospy.spin()  # If running, then let rospy.spin() handle callbacks

# 	if rospy.get_param('stop_segmentation', True):
# 		print('not got it')
# 		running = False
# 		rospy.set_param('stop_segmentation', False)  # Reset the param

# 	rospy.sleep(0.5)  # Sleep to prevent the loop from hogging CPU