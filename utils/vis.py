import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def visualize_segmentation(im, masks, nc=None, return_rgb=False, save_dir=None):



    """ Visualize segmentations nicely. Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

        @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
        @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
        @param nc: total number of colors. If None, this will be inferred by masks
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    if not return_rgb:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    if not return_rgb:
        # matplotlib stuff
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)


        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        try:
            contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        except:
            im2, contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(c.reshape((-1, 2)), fill=False, facecolor=color_mask, edgecolor='w', linewidth=1.2, alpha=0.5)
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255,255,255), 2)


    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image


def get_rects(mask):
    # Detect edges using Canny
    canny_output = cv.Canny(mask, 0, 1)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cntr in contours[::-1]:
        if cv.contourArea(cntr) < 100:
            contours.remove(cntr)
    # print(len(contours))
    rects = []
    for cntr in contours:
        area = cv.contourArea(cntr)

        if area < 100:
            continue
        # rotated bbox
        rect = cv.minAreaRect(cntr) # basically you can feed this rect into your classifier
        (x,y),(w,h), a = rect # a - angle
        

        # axis-aligned bbox
        # rect = cv.boundingRect(cntr) # basically you can feed this rect into your classifier
        # rect = np.uint0(rect)


        rects.append(rect)

    return rects


def get_rotated_rois(rgb_im, depth_im, mask):
    rects = get_rects(mask)

    warped = []
    for rect in rects:
        warped_rgb = crop_rect(rgb_im, rect)
        warped_depth = crop_rect(depth_im, rect)
        warped.append((warped_rgb, warped_depth))
 
    return warped


def crop_rect(img, rect):

    box = cv.boxPoints(rect)
    box = np.int0(box) #turn into ints    
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
  
    return warped


