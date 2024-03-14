#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO

class ROSGUI:
    def __init__(self, master):
        self.master = master
        master.title("ROS GUI")

        # Create ROS node and publishers
        rospy.init_node('ros_gui', anonymous=True)
        self.pub = rospy.Publisher('button_input', String, queue_size=10)
        self.image_sub_mask = rospy.Subscriber('/segmentation/mask', CompressedImage, self.image_callback_mask)
        self.image_sub_axis = rospy.Subscriber('/axis/image_raw/compressed', CompressedImage, self.image_callback_axis)
        self.button_input_sub = rospy.Subscriber('/button_input', String, self.button_input_callback)


        # Create buttons
        self.button1 = tk.Button(master, text="Start Segmentation", command=lambda: self.publish_input("1"))
        self.button1.pack(side=tk.TOP, expand=tk.YES)

        self.button2 = tk.Button(master, text="Stop Segmentation", command=lambda: self.publish_input("2"))
        self.button2.pack(side=tk.TOP, expand=tk.YES)

        self.button3 = tk.Button(master, text="Send Images", command=lambda: self.publish_input("3"))
        self.button3.pack(side=tk.TOP, expand=tk.YES)

        self.button4 = tk.Button(master, text="Start Annotation", command=lambda: self.publish_input("4"))
        self.button4.pack(side=tk.TOP, expand=tk.YES)

        self.button5 = tk.Button(master, text="Start FineTune", command=lambda: self.publish_input("5"))
        self.button5.pack(side=tk.TOP, expand=tk.YES)

        self.button6 = tk.Button(master, text="Send Weights", command=lambda: self.publish_input("6"))
        self.button6.pack(side=tk.TOP, expand=tk.YES)

        # Create image viewers
        self.image_label_mask = tk.Label(master)
        self.image_label_mask.pack(side=tk.RIGHT, expand=tk.YES)

        self.image_label_axis = tk.Label(master)
        self.image_label_axis.pack(side=tk.LEFT, expand=tk.YES)

        # Create terminal-like display for messages
        self.terminal_text = tk.Text(master, height=10, width=50, state=tk.DISABLED)
        self.terminal_text.pack(side=tk.BOTTOM)

    def publish_input(self, button_name):
        # Publish the button name to the 'button_input' topic
        self.pub.publish(button_name)

    def image_callback_mask(self, data):
        # Callback function for handling incoming compressed images from /segmentation/mask
        image_data = data.data
        image = Image.open(BytesIO(image_data))
        image = ImageTk.PhotoImage(image)
        self.image_label_mask.config(image=image)
        self.image_label_mask.image = image  # Keep a reference to avoid garbage collection

    def image_callback_axis(self, data):
        # Callback function for handling incoming compressed images from /axis/image_raw/compressed
        image_data = data.data
        image = Image.open(BytesIO(image_data))
        image = ImageTk.PhotoImage(image)
        self.image_label_axis.config(image=image)
        self.image_label_axis.image = image  # Keep a reference to avoid garbage collection

    def button_input_callback(self, data):
        # Callback function for handling incoming messages from /button_input
        message = data.data
        self.display_message(f"{message}")

    def display_message(self, message):
        # Display a message in the terminal-like text widget
        self.terminal_text.config(state=tk.NORMAL)
        # print(message)
        if message == "1":
            display_message = "Segmentation Started"
        elif message == "2":
            display_message = "Segmentation Stoped"
        elif message == "3":
            display_message = "Segmentation Stoped"
        elif message == "4":
            display_message = "Annotation Started"
        else:
            display_message = message
        self.terminal_text.insert(tk.END, display_message + '\n')
        self.terminal_text.config(state=tk.DISABLED)
        self.terminal_text.yview(tk.END)

if __name__ == "__main__":
    # Create the Tkinter root window
    root = tk.Tk()

    # Create an instance of the ROSGUI class
    ros_gui = ROSGUI(root)

    # Run the Tkinter event loop
    root.mainloop()
