rgbd_deep_knn
## Requirements
1.Pytorch\
2.ROS melodic\
3.cv bridge for ROS\
4.opencv\
5.Realsense plugin for ROS
## Run
In different terminals:\
roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera\
python back_subs.py\
python process_seg_results_ros_node.py\
python command_node.py\

For visualization:\
rviz rviz -d segmentation.rviz
