#!/bin/bash


rosbag record /camera/aligned_depth_to_color/image_raw /camera/color/image_raw /seg_image /class_image \
-O bags/topics.bag -j