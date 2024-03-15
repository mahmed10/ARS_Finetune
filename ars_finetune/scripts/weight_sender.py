#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import rospkg
import base64


weight_send = False
def send_file(filepath):
    with open(filepath, 'rb') as file:
        content = base64.b64encode(file.read()).decode('utf-8')
    return content

def button_input_callback(data):
    # Callback function for handling incoming messages from /button_input
    message = data.data
    global weight_send
    if message =="6":
        weight_send = True


if __name__ == "__main__":
    rospy.init_node('weight_sender', anonymous=True)
    pub = rospy.Publisher('weight_transfer', String, queue_size=100)
    button_input_sub = rospy.Subscriber('/button_input', String, button_input_callback)
    key_pub = rospy.Publisher('button_input', String, queue_size=10)


    while not rospy.is_shutdown():
        if not weight_send:
            continue

        key_pub.publish("Weight sending started")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ars_finetune')
        filepath = package_path + '/src/ars_finetune/checkpoints/model_umbc.pth'


        ### filepath = rospy.get_param('~filepath')
        rate = rospy.Rate(1)
        rate.sleep()

        rospy.loginfo("Sending file: %s", filepath)
        content = send_file(filepath)
        pub.publish(content)

        weight_send = False