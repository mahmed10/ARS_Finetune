#!/usr/bin/env python
import subprocess
import pyautogui
import time
import os

import rospy
from std_msgs.msg import String
annotation = False

def button_input_callback(data):
	# Callback function for handling incoming messages from /button_input
	message = data.data
	# print(message)
	global annotation
	if message =="4":
		annotation = True

if __name__ == "__main__":
	# Path to your AppImage
	rospy.init_node('annotation', anonymous=True)
	button_input_sub = rospy.Subscriber('/button_input', String, button_input_callback)
	while not rospy.is_shutdown():
		# print(annotation)
		if not annotation:
			continue
		# print(__file__)
		# print(os.getcwd())
		appimage_path = __file__.split('scripts')[0]+"PixelAnnotationTool_x86_64_v1.3.2.AppImage"

		# Run the AppImage
		subprocess.run([appimage_path])
		annotation = False
		# annotation = False