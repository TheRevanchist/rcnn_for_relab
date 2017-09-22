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

rand_seed = 1024

repo_of_ground_truth = '/home/revan/VOCdevkit/VOC2007/Annotations'
repo_of_images = '/home/revan/VOCdevkit/VOC2007/JPEGImages'
train_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
test_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

classes_dict = {'__background__': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
           'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14,
           'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

R_initial = np.zeros((21, 21))

def main():
    content_train = get_filenames_of_training_set(test_set)
    final_data_structure_train = create_p_and_gt(content_train, classes_dict, R_initial)
    with open('data_structure_test_peak.pickle', 'wb') as f:
        # Pickle p data structure using the highest protocol available.
        pickle.dump(final_data_structure_train, f, pickle.HIGHEST_PROTOCOL)

    # content_test = get_filenames_of_training_set(test_set)
    # final_data_structure_test = create_p_and_gt(content_test, classes_dict, R_initial)
    # with open('data_structure_test.pickle', 'wb') as f:
    #     # Pickle p data structure using the highest protocol available.
    #     pickle.dump(final_data_structure_test, f, pickle.HIGHEST_PROTOCOL)
    R = R_initial[1:,1:]
    R /= np.max(R)
    np.save('R.npy', R)
    print()


def get_filenames_of_training_set(train_set):
    """
    This function reads the names of the files that we are going to use as our training test
    :param train_set - the file which contains the names of the training set
    :return: content - an array containing the names of each filew
    """
    # read the names of the files that we are going to use as our training test
    with open(train_set) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def find_objects_in_files(filename, dict):
    """
    This function finds the centers of each object in the image
    :param element: the name of the image
    :return: points - a two 2 array containing the centers of each bounding box
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    gt_info = []

    for size in root.iter('size'):
        for width in size.iter('width'):
            width = float(int(width.text))
        for height in size.iter('height'):
            height = float(int(height.text))

    for object in root.iter('object'):
        for name in object.iter('name'):
            class_of_object = name.text
            gt_info_object = np.zeros(25)
            if class_of_object in dict:  # skip parts of bodies (head, foot, hand etc)
                name_decrypted = dict[class_of_object]
                gt_info_object[name_decrypted] = 1.0

        for bounding_box in object.iter('bndbox'):
            for xmin in bounding_box.iter('xmin'):
                # we convert first to float and then to int because for some reasons the bounding boxes can have float
                # coordinates like 273.3863
                xmin = int(float(xmin.text)) # / width
            for ymin in bounding_box.iter('ymin'):
                ymin = int(float(ymin.text)) # / height
            for xmax in bounding_box.iter('xmax'):
                xmax = int(float(xmax.text)) # / width
            for ymax in bounding_box.iter('ymax'):
                ymax = int(float(ymax.text)) # / height
            # fill the spatial positions for each object
            gt_info_object[21] = xmin
            gt_info_object[22] = ymin
            gt_info_object[23] = xmax
            gt_info_object[24] = ymax
        gt_info.append(gt_info_object)
    gt_info = np.asarray(gt_info)
    return gt_info, width, height


def create_data_structure(filename, dict, R):
    gt_info, width, height = find_objects_in_files(filename, dict)
    _argmax = np.argmax(gt_info[:, :21], axis=1)
    for i in xrange(len(_argmax)):
        for j in xrange(len(_argmax)):
            R[_argmax[i], _argmax[j]] += 1
    #p = np.random.rand(len(gt_info), 20)
    #p /= np.sum(p, axis=1)[:, np.newaxis]

    peak_probability = 0.5 # this should be put as a function argument
    prob_of_others = (1 - peak_probability) / 20
    p = np.full((len(gt_info), 20), prob_of_others)
    peak_indexes = np.argmax(gt_info[:, 1:21], axis=1)
    p[:, peak_indexes] = peak_probability
    p[:, peak_indexes] = peak_probability

    gt = gt_info[:, 1:21]
    p_rect = gt_info[:, 21:]
    gt_rect = p_rect
    width_and_height = np.asarray([width, height])
    all_data_structure = []
    all_data_structure.append(p)
    all_data_structure.append(gt)
    all_data_structure.append(p_rect)
    all_data_structure.append(gt_rect)
    all_data_structure.append(width_and_height)
    return all_data_structure


def create_p_and_gt(content, dict, R):

    final_data_structure = []
    i = 0
    for image_name in content:
        xml_file = os.path.join(repo_of_ground_truth, image_name + ".xml")
        all_data_structure = create_data_structure(xml_file, dict, R)
        final_data_structure.append(all_data_structure)
        i += 1
        if i % 100 == 0:
            print(str(i) + " images processed")

    return final_data_structure


if __name__ == "__main__":
    main()
    print("done")