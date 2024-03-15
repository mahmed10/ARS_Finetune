# ARS_Finetune

## Prerequisite 
ROS Noetic in the both Server and Edge node. To install ROS Noetic please follow thes steps: 
[ROS Noetic Installation Steps](http://wiki.ros.org/noetic/Installation)

## Installation The framework
1. Create a catkin workspace `mkdir -p catkin_ws/src`
2. Navigate to src folder `cd catkin_ws/src`
3. Clone the repo in the src folder `git clone`
4. Navigate to catkin_ws folder `cd ..`
5. Build the project with catkin_make `catkin_make`


## Run the Code
Open terminal, and type the following command
1. For Server `roslaunch ars_finetune server.launch`
2. For Edge `roslaunch ars_finetune robot.launch`
