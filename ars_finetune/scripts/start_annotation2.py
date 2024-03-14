#!/usr/bin/env python
# import subprocess
import pyautogui
import time
import keyboard

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
	rospy.init_node('annotation2', anonymous=True)
	button_input_sub = rospy.Subscriber('/button_input', String, button_input_callback)
	while not rospy.is_shutdown():
		# print(annotation)
		if not annotation:
			continue
		# Click the "File" menu (you might need to adjust the coordinates)
		time.sleep(2)
		pyautogui.click(x=143, y=73)  # x, y are the coordinates of the "File" menu

		# Wait for the dropdown to appear
		time.sleep(1)

		# Click the "Open Directory" option (again, adjust the coordinates)
		pyautogui.click(x=148, y=89)  # x, y are the coordinates of the "Open Directory" option


		# Wait for the dropdown to appear
		time.sleep(2)

		pyautogui.write('/home/mpsc/masud_ws/src/ars_finetune/src/ars_finetune/output_file/images')

		time.sleep(1)

		pyautogui.press('enter')
		annotation = False
		# annotation = False