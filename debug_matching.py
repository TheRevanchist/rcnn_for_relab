import numpy as np
import os
import xml.etree.ElementTree as ET
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
import pickle
from numpy import unravel_index
from test2 import *
from input_relab import get_filenames_of_set as files, bb_intersection_over_union as iou


repo_of_ground_truth = '/home/revan/VOCdevkit/VOC2007/Annotations'
repo_of_images = '/home/revan/VOCdevkit/VOC2007/JPEGImages'
train_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
test_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/test.txt'


def main():
    debug_bounding_boxes(repo_of_images)


def debug_bounding_boxes(repo_of_images):
    """
    This function visualizes the images, by putting bounding boxes with the same color on rcnn boxes and ground truth
    boxes, in addition to printing the results (iou) in the console
    :param repo_of_images: the repository of the images
    """
    content = files(train_set)
    with open('info_all_images_trainval.pickle', 'rb') as f:
        info_all_images = pickle.load(f)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, im in enumerate(content):
        # read the image in opencv
        image = cv2.imread(os.path.join(repo_of_images, im + ".jpg"))

        # get all the information about the image
        current_image = info_all_images[i]
        p_rect_image = current_image[2]
        gt_rect_image = current_image[3]

        # get number of objects in the image
        number_of_objects = p_rect_image.shape[0]

        # get a list of distinct colors to draw each object
        distinct_colors = get_spaced_colors(number_of_objects)

        for j in xrange(number_of_objects):
            print(p_rect_image[j], gt_rect_image[j], iou(p_rect_image[j], gt_rect_image[j]))
            cv2.rectangle(image, (int(p_rect_image[j, 0]), int(p_rect_image[j, 1])),
                          (int(p_rect_image[j, 2]), int(p_rect_image[j, 3])), distinct_colors[j], thickness=1, lineType=4)
            cv2.putText(image, str(j), (int(p_rect_image[j, 0]), int(p_rect_image[j, 1])), font, 1, distinct_colors[0], 1, cv2.LINE_AA)
            cv2.rectangle(image, (int(gt_rect_image[j, 0]), int(gt_rect_image[j, 1])),
                          (int(gt_rect_image[j, 2]), int(gt_rect_image[j, 3])), distinct_colors[j], thickness=1, lineType=8)
            cv2.putText(image, str(j), (int(gt_rect_image[j, 2]), int(gt_rect_image[j, 3])), font, 1, distinct_colors[0],
                        1, cv2.LINE_AA)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        print("\n\n\n\n\n\n")


def get_spaced_colors(n):
    """
    This function creates a list of n distinct colors
    :param n: the number of colors
    :return: a list of distinct colors
    """
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


if __name__ == "__main__":
    main()