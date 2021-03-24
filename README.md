## Requirements
- Pytorch
- ROS melodic
- cv bridge for ROS
- opencv
- Realsense plugin for ROS
## Run
In different terminals:

- Run realsense-ros connection
  - roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera\
- Run segmentation bt background substruction
  - python back_subs.py\
- Run knn node:
  - python process_seg_results_ros_node.py\
- Run node from which we will send commands:
  - python command_node.py\


For visualization:
- rviz rviz -d segmentation.rviz
