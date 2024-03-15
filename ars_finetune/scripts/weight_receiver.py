#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import rospkg
import base64
import os

def callback(data):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('ars_finetune') + '/src/ars_finetune/checkpoints/'
    file_path = package_path + 'model_umbc_new.pth'


    with open(file_path, 'wb') as file:
        decoded_file = base64.b64decode(data.data)
        file.write(decoded_file)
    rospy.loginfo("Received and saved weight file")
    notification.publish("recieved weight file")


if __name__ == "__main__":
    # Create the directory
    # The 'exist_ok' parameter is set to True, which means the function won't raise an error if the directory already exists.
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('ars_finetune')

    rospy.init_node('weight_receiver', anonymous=True)
    rospy.Subscriber('weight_transfer', String, callback)
    notification = rospy.Publisher('weight_recive_notification', String, queue_size=10)
    rospy.spin()