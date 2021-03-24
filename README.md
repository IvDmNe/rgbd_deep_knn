## Requirements
- Pytorch
- ROS melodic
- cv bridge for ROS
- opencv
- Realsense plugin for ROS
## Run
In different terminals:
- roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera
- python back_subs.py
- python process_seg_results_ros_node.py
- python command_node.py


For visualization:
- rviz rviz -d segmentation.rviz
