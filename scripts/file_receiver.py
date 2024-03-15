#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import rospkg
import base64
import os

received_txt = False  # Flag to keep track of which file type is being processed
image_index = 0
total_img = 0
image_name = []


def callback(data):
    global received_txt
    global image_index
    global total_img
    global image_name
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('ars_finetune') + '/src/ars_finetune/output_file/'
    file_path = package_path + 'info.txt'
    os.makedirs(package_path + 'images', exist_ok=True)

    if received_txt == False:
        with open(file_path, 'w') as file:
            file.write(data.data)
        received_txt = True
        rospy.loginfo("Received and saved file: %s", file_path)
        notification.publish("recieved txt file")

        with open(file_path, 'r') as file:
            lines = file.readlines()
        image_name = sorted(lines, key=lambda line: float(line.split(",")[0]), reverse = True)
        total_img = len(image_name)
    
    else:
        file_path = package_path+'images/'+ image_name[image_index].split(',')[1]
        with open(file_path, 'wb') as file:
            decoded_file = base64.b64decode(data.data)
            file.write(decoded_file)
        rospy.loginfo("Received and saved file: %s", file_path)
        notification.publish(image_name[image_index])
        image_index += 1

        if image_index == total_img:
            received_txt = False  # Flag to keep track of which file type is being processed
            image_index = 0

if __name__ == "__main__":
    # Create the directory
    # The 'exist_ok' parameter is set to True, which means the function won't raise an error if the directory already exists.
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('ars_finetune')

    rospy.init_node('file_receiver', anonymous=True)
    rospy.Subscriber('file_transfer', String, callback)
    notification = rospy.Publisher('recive_notification', String, queue_size=10)
    rospy.spin()