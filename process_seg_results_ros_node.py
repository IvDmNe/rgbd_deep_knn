#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test UCN on ros images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import tf
import rosnode
import message_filters
import cv2
import torch.nn as nn
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import _init_paths
import networks
import rospy
import copy

from sensor_msgs.msg import Image, CameraInfo, String

from cv_bridge import CvBridge, CvBridgeError
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.test_dataset import test_sample
from utils.mask import visualize_segmentation

from models.knn_classifier import knn_torch
from models.feature_extractor import feature_extractor
from utils.vis import get_rotated_rois
from models.classifier import classifier


lock = threading.Lock()




def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


class ImageListener:

    def __init__(self):

  

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None


        self.classifier = classifier()

        self.save_dir = 'save_images'
        os.makedirs(self.save_dir, exists_ok=True)

        # initialize a node
        rospy.init_node("classification")
        self.label_pub = rospy.Publisher('rgb_with_labels', Image, queue_size=10)
        # self.label_refined_pub = rospy.Publisher('seg_label_refined', Image, queue_size=10)
        # self.image_pub = rospy.Publisher('seg_image', Image, queue_size=10)
        # self.image_refined_pub = rospy.Publisher('seg_image_refined', Image, queue_size=10)
        # self.feature_pub = rospy.Publisher('seg_feature', Image, queue_size=10)


        self.base_frame = 'measured/base_link'
        rgb_sub = message_filters.Subscriber('/realsense_plugin/camera/color/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/realsense_plugin/camera/depth/image_raw', Image, queue_size=10)
        mask_sub = message_filters.Subscriber('/seg_rgb/seg_label', Image, queue_size=10)
        msg = rospy.wait_for_message('/realsense_plugin/camera/color/camera_info', CameraInfo)
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame
 

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, mask_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        rospy.Subscriber('/classification/commands', String, self.callback_working_mode)


    def callback_working_mode(self, command):

        if ' ' in command and (command[0] == 'train'):

            self.mode = command[0]
            self.train_cl = command[1]
        elif command == 'inference':
            self.mode = 'inference'

        print('|||||||||||||||||||||||||||||||')
        print('mode changed to', command)
        print('|||||||||||||||||||||||||||||||')
        

    def callback_rgbd(self, rgb, depth, mask):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            # depth_cv /= 1000.0

            mask_cv = self.cv_bridge.imgmsg_to_cv2(mask)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        # rescale image if necessary
        if cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.mask = mask_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_proc(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            mask_img = self.mask()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===========================================')


        # rgb_rois, depth_rois, rects = get_rotated_rois(im_color, depth_img, mask_img)
   
        if self.classifier.mode == 'train':

            self.classifier.save_rois(im_color, depth_img, mask_img, self.train_cl)
            self.classifier.was_trained = True
    

        elif self.classifier.mode == 'inference':
            
            #  at the end of training get deep features from images and save them to file
            if self.classifier.was_trained:
                self.classifier.save_deep_features()
                self.classifier.was_trained = False

            # feed to feature extractor each roi
            rgb_rois, depth_rois, rects = get_rotated_rois(im_color, depth_img, mask_img)
            rgb_rois = convert_to_tensor(rgb_rois, shape=(224, 224))
            depth_rois = convert_to_tensor(depth_rois, shape=(224, 224))

            deep_rgb_features = self.classifier.extractor(rgb_rois)
            deep_depth_features = self.classifier.extractor(depth_rois)

            deep_features = torch.cat([deep_rgb_features, deep_depth_features], dim=1)

            classes = self.classifier.knn.classify(deep_features)

            print(classes)
            



             



        # # 
        # bgr image
        # im = im_color.astype(np.float32)
        # im_tensor = torch.from_numpy(im) / 255.0
        # pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        # im_tensor -= pixel_mean
        # image_blob = im_tensor.permute(2, 0, 1)
        # sample = {'image_color': image_blob.unsqueeze(0)}

        # if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        #     height = im_color.shape[0]
        #     width = im_color.shape[1]
        #     xyz_img = compute_xyz(depth_img, self.fx, self.fy, self.px, self.py, height, width)
        #     depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        #     sample['depth'] = depth_blob.unsqueeze(0)




        # out_label, out_label_refined = test_sample(sample, self.network, self.network_crop)

        # # publish segmentation mask
        # label = out_label[0].cpu().numpy()
        # label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        # label_msg.header.stamp = rgb_frame_stamp
        # label_msg.header.frame_id = rgb_frame_id
        # label_msg.encoding = 'mono8'
        # self.label_pub.publish(label_msg)

        # num_object = len(np.unique(label)) - 1
        # print('%d objects' % (num_object))

        # if out_label_refined is not None:
        #     label_refined = out_label_refined[0].cpu().numpy()
        #     label_msg_refined = self.cv_bridge.cv2_to_imgmsg(label_refined.astype(np.uint8))
        #     label_msg_refined.header.stamp = rgb_frame_stamp
        #     label_msg_refined.header.frame_id = rgb_frame_id
        #     label_msg_refined.encoding = 'mono8'
        #     self.label_refined_pub.publish(label_msg_refined)

        # # publish segmentation images
        # im_label = visualize_segmentation(im_color[:, :, (2, 1, 0)], label, return_rgb=True)
        # rgb_msg = self.cv_bridge.cv2_to_imgmsg(im_label, 'rgb8')
        # rgb_msg.header.stamp = rgb_frame_stamp
        # rgb_msg.header.frame_id = rgb_frame_id
        # self.image_pub.publish(rgb_msg)

        # if out_label_refined is not None:
        #     im_label_refined = visualize_segmentation(im_color[:, :, (2, 1, 0)], label_refined, return_rgb=True)
        #     rgb_msg_refined = self.cv_bridge.cv2_to_imgmsg(im_label_refined, 'rgb8')
        #     rgb_msg_refined.header.stamp = rgb_frame_stamp
        #     rgb_msg_refined.header.frame_id = rgb_frame_id
        #     self.image_refined_pub.publish(rgb_msg_refined)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = args.instance_id
    num_classes = 2
    cfg.MODE = 'TEST'
    cfg.TEST.VISUALIZE = False
    print('GPU device {:d}'.format(args.gpu_id))

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        network_crop.eval()
    else:
        network_crop = None

    # image listener
    listener = ImageListener(network, network_crop)
    while not rospy.is_shutdown():
       listener.run_network()
