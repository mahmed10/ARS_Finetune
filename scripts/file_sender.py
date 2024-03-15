#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import rospkg
import time
import base64
import shutil

file_send = False
recieve_notice = ''
def send_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    return content

def send_img(filepath):
    with open(filepath, 'rb') as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string

def button_input_callback(data):
    # Callback function for handling incoming messages from /button_input
    message = data.data
    global file_send
    if message =="3":
        file_send = True

def recieve_notice_callback(data):
    global recieve_notice
    recieve_notice = data.data

if __name__ == "__main__":
    rospy.init_node('file_sender', anonymous=True)
    pub = rospy.Publisher('file_transfer', String, queue_size=10)
    button_input_sub = rospy.Subscriber('/button_input', String, button_input_callback)
    key_pub = rospy.Publisher('button_input', String, queue_size=10)
    notification = rospy.Subscriber('/recive_notification', String, recieve_notice_callback)
    # global file_send
    # print(file_send)
    # rospy.spin()
    while not rospy.is_shutdown():
        # print(file_send)

        if not file_send:
            continue

        key_pub.publish("File sending started")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ars_finetune')
        filepath = package_path + '/src/ars_finetune/output_file/info.txt'

        with open(filepath, 'r') as file:
            lines = file.readlines()
        # # Sort the lines based on the entropy value (which is the first value on each line)
        sorted_lines = sorted(lines, key=lambda line: float(line.split(",")[0]), reverse = True)
        sorted_lines = sorted_lines[:int(len(sorted_lines)*0.2)+1]

        # # Write the sorted lines back to the file
        with open(filepath, 'w') as file:
            file.writelines(sorted_lines)

        ### filepath = rospy.get_param('~filepath')
        rate = rospy.Rate(10)
        rate.sleep()

        rospy.loginfo("Sending file: %s", filepath)
        content = send_file(filepath)
        pub.publish(content)
        while(1):
            if recieve_notice == 'recieved txt file':
                break

        key_pub.publish("Sending "+str(len(sorted_lines))+" Files")
        for i in sorted_lines:
            filepath = package_path + '/src/ars_finetune/output_file/images/' + i.split(',')[1]
            key_pub.publish("Sending "+ i.split(',')[1])

            rospy.loginfo("Sending file: %s", filepath)
            content = send_img(filepath)
            pub.publish(content)
            print(i)
            while(1):
                if recieve_notice == i:
                    break
        key_pub.publish("File sending done")

        try:
            shutil.rmtree(package_path + '/src/ars_finetune/output_file/')
        except Exception as e:
            rospy.loginfo(f"Error deleting {directory_path}: {e}")

        key_pub.publish("1")

        file_send = False  