#!/usr/bin/env python3
import rospy
import subprocess
import sys
import time

def measure_power_consumption(python_script_path, duration_seconds):
    try:
        # Start powerstat in background
        powerstat_process = subprocess.Popen(['powerstat', '-d', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Run the Python ROS node
        ros_node_process = subprocess.Popen(['python3', python_script_path])

        # Wait for the duration specified
        time.sleep(duration_seconds)

        # Terminate the ROS node after the duration
        ros_node_process.terminate()

        # Terminate powerstat after the duration
        powerstat_process.terminate()

        # Gather powerstat output
        stdout, stderr = powerstat_process.communicate()
        print(stdout.decode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # script_path = sys.argv[1]
    script_path = '/home/mpsc/masud_ws/src/ars_finetune/src/ars_finetune/segmentation.py'
    measure_power_consumption(script_path, 40)
