

import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
from matplotlib import pyplot as plt
from torch import nn
from utils.vis import visualize_segmentation, get_rects, get_rotated_rois
from utils.augment import convert_to_tensor
from models.feature_extractor import feature_extractor


def get_compactness(cntrs):

    if isinstance(cntrs, list):
        compactness = []

        for cntr in cntrs:
            area = cv.contourArea(cntr)
            perimeter = cv.arcLength(cntr, True)
            compactness.append((perimeter**2) / area)
        return np.array(compactness)
    else:

        area = cv.contourArea(cntrs)
        perimeter = cv.arcLength(cntrs, True)
        compactness = ((perimeter**2) / area)
        return compactness


def get_centers(cntrs):
    if isinstance(cntrs, list):
        centers = []

        for cntr in cntrs:
            M = cv.moments(cntr)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        return np.array(centers)
       
    else:
        M = cv.moments(cntrs)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers = [cX, cY]
        return np.array(centers)

def find_nearest(array, value):

    array = np.asarray(array)
    distances = np.linalg.norm((array - value), axis=1)
    
    return np.min(distances), np.argmin(distances)

class useless:

    def __init__(self):
        self.prev_cntrs = None
        self.prev_ids = None

        self.prev_cntrs_comp = None
        self.prev_cntrs_centers = None

        self.total_id = 0

    def assign_id(self, cntrs):

        if self.total_id == 0:
            ret_ids = list(range(self.total_id, len(cntrs)))
            self.total_id += len(cntrs)
            # self.prev_cntrs = cntrs
            self.prev_ids = ret_ids
            # self.prev_cntrs_comp = get_compactness(cntrs)
            self.prev_cntrs_centers = get_centers(cntrs)

            return ret_ids


        cur_ids = [None]*len(cntrs)

        centers = []

        for idx, cntr in enumerate(cntrs):

            # cur_comp = get_compactness(cntr)
            cur_center = get_centers(cntr)
            

            # print(self.prev_cntrs_comp)
            dist, nearest = find_nearest(self.prev_cntrs_centers, cur_center)
            

            if (dist < 50):

                cur_ids[idx] = self.prev_ids[nearest]
                self.prev_cntrs_centers[nearest] = cur_center

            else:
                cur_ids[idx] = self.total_id
                self.total_id += 1
                # print(self.prev_cntrs_centers, cur_center)
                centers.append(cur_center)
        # print(centers)
        if centers:
            self.prev_cntrs_centers = np.concatenate([self.prev_cntrs_centers, centers])
        # print(self.prev_cntrs_centers)
        self.prev_ids += cur_ids
        # self.prev_cntrs_comp = np.concatenate([self.prev_cntrs_comp, comps])
        
        return cur_ids

            

            

            

        

            



if __name__ =='__main__':

    prev_rects = None

    total_id = 0

    u = useless()

    for i in range(58):
        print()

        print(i, '    -----------')

        rgb = cv.imread(f'out_images/rgb_img{i}.png')
        depth_im = cv.imread(f'out_images/depth_img{i}.png')
        depth = cv.cvtColor(depth_im, cv.COLOR_BGR2GRAY)
        mask = cv.imread(f'out_images/mask{i}.png')
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)


        rgb_rois, depth_rois, cntrs = get_rotated_rois(rgb, depth, mask)

        if not rgb_rois:
            print('no objects')
            continue        

        
        drawing = rgb.copy()

        cv.drawContours(drawing, cntrs, -1, (255, 0, 255), 2)

        
        center_cntr = find_nearest_to_center_cntr(cntrs, rgb.shape)

        

        if center_cntr is not None:
            cv.drawContours(drawing, [center_cntr], -1, (255, 0, 0), 2)

            M = cv.moments(center_cntr)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(drawing, (cX, cY), 2, (255,0,0))

            center_roi = cntrs.index(center_cntr)
            center_roi = rgb_rois[center_roi]
            plt.imshow(center_roi)
            plt.show()


        cv.circle(drawing, (rgb.shape[1] //2, rgb.shape[0] //2), 200, (0,255,0))
        cv.circle(drawing, (rgb.shape[1] //2, rgb.shape[0] //2), 2, (0,0,255))

        # cv.putText(drawing, str(idx), (cX - 10, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        plt.imshow(drawing)
        plt.show()




        # print(u.prev_cntrs_comp)
        # cur_ids = u.assign_id(cntrs)

        # print('total id',u.total_id)
        # print('rois', len(rgb_rois))
        # print('cur id', cur_ids)

        # drawing = rgb.copy()
        # for cntr, idx in zip(cntrs, cur_ids):
        #     cv.drawContours(drawing, [cntr], -1, (255, 0, 0), 2)

        #     M = cv.moments(cntr)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])

        #     cv.putText(drawing, str(idx), (cX - 10, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        # plt.imshow(drawing)
        # plt.show()
        # print(len(cntrs))



        # f= plt.figure()
        # f.add_subplot(1,2, 1)
        # plt.imshow(rgb)
        # f.add_subplot(1,2, 2)
        # plt.imshow(mask)
        # plt.show(block=True)


        





   