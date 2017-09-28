# This files creates all the needed data structures that relab needs
# User gives the option for training and testing mode

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
from faster_rcnn.nms.py_cpu_nms import py_cpu_nms

repo_of_ground_truth = '/home/revan/VOCdevkit/VOC2007/Annotations'
repo_of_images = '/home/revan/VOCdevkit/VOC2007/JPEGImages'
train_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
test_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/test.txt'


classes_dict = {'__background__': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
           'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14,
           'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

imdb_name = 'voc_2007_test'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
trained_model = 'VGGnet_fast_rcnn_iter_70000.h5'
rand_seed = 1024
save_name = 'faster_rcnn_100000'
max_per_image = 300
thresh = 0.05
vis = False


def main(train=1, serialize=1):
    distribution_mode = 'k-peak'  # the mode can be 'peak', k_peak or 'softmax'
    if train:
        if serialize:
            content = get_filenames_of_set(train_set)
            p, gt, width_and_height = create_p_and_gt(content, repo_of_images, classes_dict, distribution_mode)

            # dump data structures into pickle files
            with open('p_trainval.pickle', 'wb') as f:
                pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
            with open('gt_trainval.pickle', 'wb') as f:
                pickle.dump(gt, f, pickle.HIGHEST_PROTOCOL)
            with open('width_and_height_traival.pickle', 'wb') as f:
                pickle.dump(width_and_height, f, pickle.HIGHEST_PROTOCOL)

        # load the pickle files
        with open('p_trainval.pickle', 'rb') as f:
            p = pickle.load(f)
        with open('gt_trainval.pickle', 'rb') as f:
            gt = pickle.load(f)
        with open('width_and_height_traival.pickle', 'rb') as f:
            width_and_height = pickle.load(f)

        if serialize:
            # do the matching between rcnn results and the ground truth
            info_all_images = postprocess_all_images(p, gt, width_and_height, false_positives=False, false_negatives=False)

            # pickle the final data structure
            with open('train_k_peak_nofp_nofn.pickle', 'wb') as f:
                pickle.dump(info_all_images, f, pickle.HIGHEST_PROTOCOL)

        # load the final data structure
        with open('train_k_peak_nofp_nofn.pickle', 'rb') as f:
            info_all_images = pickle.load(f)

    else:
        if serialize:
            content = get_filenames_of_set(test_set)
            p, gt, width_and_height = create_p_and_gt(content, repo_of_images, classes_dict, distribution_mode)

            # dump data structures into pickle files
            with open('p_test.pickle', 'wb') as f:
                pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
            with open('gt_test.pickle', 'wb') as f:
                pickle.dump(gt, f, pickle.HIGHEST_PROTOCOL)
            with open('width_and_height_test.pickle', 'wb') as f:
                pickle.dump(width_and_height, f, pickle.HIGHEST_PROTOCOL)

        # load the pickle files
        with open('p_test.pickle', 'rb') as f:
            p = pickle.load(f)
        with open('gt_test.pickle', 'rb') as f:
            gt = pickle.load(f)
        with open('width_and_height_test.pickle', 'rb') as f:
            width_and_height = pickle.load(f)

        if serialize:
            info_all_images = []
            len_p = len(p)
            for i in xrange(len_p):
                new_data = []
                if len(p[i] > 0):
                    new_p = p[i][:, :21]
                    new_rect_p = p[i][:, 21:]

                else:
                    new_p = []
                    new_rect_p = []

                new_gt = gt[i][:, :21]
                new_rect_gt = gt[i][:, 21:]
                new_width_and_height = width_and_height[i]
                new_data.append(new_p)
                new_data.append(new_gt)
                new_data.append(new_rect_p)
                new_data.append(new_rect_gt)
                new_data.append(new_width_and_height)
                info_all_images.append(new_data)

            with open('info_all_images_test.pickle', 'wb') as f:
                pickle.dump(info_all_images, f, pickle.HIGHEST_PROTOCOL)

        # load the final data structure
        with open('info_all_images_test.pickle', 'rb') as f:
            info_all_images = pickle.load(f)

    print("Done")


def remove_background(p):
    """
    This function removes the background class, by spreading its probability over all other classes
    :param p: representation of the class as a probability function (final 4 elements are spatial dimensions)
    :return:
            new_p: representation of the class as a probability function with background removed
            new_rect: the bounding box
    """
    new_p = []
    new_rect = []
    for i in xrange(len(p)):
        if len(p[i]) > 0:
            probs = p[i][:, 1:21]
            probs /= np.sum(probs, axis=1)[:, np.newaxis]
            rects = p[i][:, 21:]
            new_p.append(probs)
            new_rect.append(rects)
        else:
            new_p.append(np.zeros((0, 0)))
            new_rect.append(np.zeros((0, 0)))
    return new_p, new_rect



def get_filenames_of_set(train_set):
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
                xmin = int(float(xmin.text))  # / width
            for ymin in bounding_box.iter('ymin'):
                ymin = int(float(ymin.text))  # / height
            for xmax in bounding_box.iter('xmax'):
                xmax = int(float(xmax.text))  # / width
            for ymax in bounding_box.iter('ymax'):
                ymax = int(float(ymax.text))  # / height
            # fill the spatial positions for each object
            gt_info_object[21] = xmin
            gt_info_object[22] = ymin
            gt_info_object[23] = xmax
            gt_info_object[24] = ymax
        gt_info.append(gt_info_object)
    gt_info = np.asarray(gt_info)
    return gt_info, width, height


def test_net(name, net, imdb, im, max_per_image=300, thresh=0.3, vis=False):
    """Test a Fast R-CNN network on an image database."""

    scores, boxes = im_detect(net, im)

    to_keep = []

    all_class_boxes = []

    for j in xrange(1, imdb.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        object_class = np.full((cls_dets.shape[0], 1), j, dtype=np.float32)
        cls_dets = np.hstack((cls_dets, object_class))#.astype(np.float32, copy=False)
        cls_dets = cls_dets[keep, :]
        if cls_dets.shape[0] != 0:
            for k in xrange(len(cls_dets)):
                all_class_boxes.append(cls_dets[k])

        to_keep.extend(inds[keep])

    all_scores = scores[to_keep]
    class_boxes = np.zeros((len(all_class_boxes), 6))
    for whatever in range(len(all_class_boxes)):
        class_boxes[whatever] = all_class_boxes[whatever]
    return all_scores, class_boxes


def create_p_and_gt(content, repo_of_images, dict, mode='peak'):
    """
    This function creates initial p and gt based on the output of the r-cnn and the ground truth
    :param content: a list containing the names of the images
    :param repo_of_images: the repository of the images
    :param dict: a dictionary where as key are names of the classes, and as values are an enumeration of them
    :return:
            rcnn_output: the representation given by rcnn
            ground_truth: the representation gotten from the ground truth
            all_width_and_height: width and the height of the image (read from xml files)
    """
    rcnn_output = []
    ground_truth = []
    all_width_and_height = []

    for image_name in content:
        im = cv2.imread(os.path.join(repo_of_images, image_name + ".jpg"))
        xml_file = os.path.join(repo_of_ground_truth, image_name + ".xml")

        all_scores, all_class_boxes = test_net(save_name, net, imdb, im, max_per_image, thresh=thresh, vis=0)

        all_scores_length = len(all_scores)
        if mode == 'peak':
            for i in xrange(all_scores_length):
                all_scores[i, :] = create_peak_array(all_class_boxes[i, 4], all_class_boxes[i, 5])

        elif mode == 'k-peak':
            k = 5
            for i in xrange(all_scores_length):
                all_scores[i, :] = create_peak_k_array(all_scores[i, :], k)

        gt_info, width, height = find_objects_in_files(xml_file, dict)
        ground_truth.append(gt_info)

        if len(all_scores) != 0:
            rcnn_output_individual = np.zeros((len(all_scores), 25))
            rcnn_output_individual[:, :21] = all_scores
            rcnn_output_individual[:, 21:] = all_class_boxes[:, :-2]
            rcnn_output.append(rcnn_output_individual)
            all_width_and_height.append(np.asarray([width, height]))
        else:
            rcnn_output.append(np.asarray([]))  # we add an empty array just to have everything synchronized
            all_width_and_height.append(np.asarray([width, height]))

    return rcnn_output, ground_truth, all_width_and_height


def bb_intersection_over_union(boxA, boxB):
    """
    This function does intersection over union between two bounding boxes
    :param boxA: box x1 represented as [min_x1, min_y1, max_x1, max_y1]
    :param boxB: box x2 represented as [min_x2, min_y2, max_x2, max_y2
    :return: iou: intersection over union - a number between 0 and 1
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA > xB or yA > yB:
        return 0

    else:
        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


def postprocess(p, gt, width_and_height, false_positives=False, false_negatives=False):
    """
    This function does matching and then postprocessing of p's and gt's
    :param p: the objects given from rcnn
    :param gt: the objects we get from the ground truth
    :param width_and_height: the width and height of the image
    :return: info_image: a list which contains the postprocessed p, rectangels for p, postprocessed gt, rectangles
             for gt, width and height
    """
    len_p = len(p)
    len_gt = len(gt)
    elements_in_p = [i for i in xrange(len_p)]
    elements_in_gt = [i for i in xrange(len_gt)]

    matching_table = create_matching_table(p, gt)
    max_number_of_matches = min(matching_table.shape[0], matching_table.shape[1])
    new_p = []
    new_gt = []
    new_rects_p = []
    new_rects_gt = []

    # on this part we create the real matches between p and gt
    for _ in xrange(max_number_of_matches):
        best_match = unravel_index(matching_table.argmax(), matching_table.shape)
        if matching_table[best_match[0], best_match[1]]: # check if it is a different value from 0
            matching_table[best_match[0], :] = 0.
            matching_table[:, best_match[1]] = 0.
            new_p.append(p[best_match[0], :21])
            new_rects_p.append(p[best_match[0], 21:])
            new_gt.append(gt[best_match[1], :21])
            new_rects_gt.append(gt[best_match[1], 21:])
            elements_in_p.remove(best_match[0])
            elements_in_gt.remove(best_match[1])

    # here we add the matches of false positives by inserting background class on the given rectangles on the ground
    # truth
    if false_positives:
        for element in elements_in_p:
            new_p.append(p[element, :21])
            new_rects_p.append(p[element, 21:])
            new_gt.append(create_background_peak_array())
            new_rects_gt.append(p[element, 21:])

    # here we deal with false negatives, by adding them as r-cnn outputs equal to the ground truth
    if false_negatives:
        for element in elements_in_gt:
            new_p.append(gt[element, :21])
            new_rects_p.append(gt[element, 21:])
            new_gt.append(gt[element, :21])
            new_rects_gt.append(gt[element, 21:])

    # convert all the lists to numpy arrays
    new_p = np.asarray(new_p)
    new_rects_p = np.asarray(new_rects_p)
    new_gt = np.asarray(new_gt)
    new_rects_gt = np.asarray(new_rects_gt)

    # add all the postprocessed information to a list
    info_image = [new_p, new_gt, new_rects_p, new_rects_gt, width_and_height]

    return info_image


def create_peak_array(peak, peak_index):
    """
    This function creates an array which represents a probability distribution, with value peak in the peak_index'th
    entry, and value (1 - peak)/20. in all the other positions
    :param peak: the peak value
    :param peak_index: the index where we have to put the peak value
    :return:
    """
    not_peak = (1. - peak) / 20.
    peak_distribution = np.full((1, 21), not_peak, dtype=np.float32)
    peak_distribution[0, int(peak_index)] = peak
    return peak_distribution


def create_peak_k_array(softmax_array, k):
    """
    This function thresholds to 0 all entries which are not in the top k values
    :param softmax_array: an array which represents a probability distribution over classes
           k: k peak elements
    :return: k_peak_array - the postprocessed softmax array
    """
    argsort_array = np.argsort(softmax_array)[::-1] # argsort in reverse order
    k_peak_array = np.zeros(21)
    for i in xrange(k):
        k_peak_array[argsort_array[i]] = softmax_array[argsort_array[i]]
    k_peak_array /= np.sum(k_peak_array)
    return k_peak_array



def create_matching_table(p, gt):
    """
    This function creates a table of size n by m (n number of objects in p_rect, m number of objects in gt_rect),
    where the entries on the table are the values of intersection over union of those objects
    :param p: a numpy array of size n by 25 (21 first elements are the prob. distribution, last 4 elements are sptatial
              dimensions
    :param gt_rect: same as p, but the probability distribution is taken from the ground truth
    :return: matching table: the table containing all i_over_u for each cartesian product between p and gt
    """
    len_p = len(p)
    len_gt = len(gt)
    matching_table = np.zeros((len_p, len_gt))
    for i in xrange(len_p):
        for j in xrange(len_gt):
            matching_table[i, j] = bb_intersection_over_union(p[i, -4:], gt[j, -4:])
    return matching_table


def create_background_peak_array():
    """
    This function simply returns an array with 1 in the background class and 0 in all other classes
    :return: the above mentioned array
    """
    return np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


def postprocess_all_images(p, gt, width_and_height):
    """
    This function iterates over all images, calling postprocess
    :param p: data structure containing information given from rcnn
    :param gt: data structure containing information given from the ground truth
    :param width_and_height: width and height info for all images
    :return: info_all_images: a list containing all information needed from relab
    """
    number_of_images = len(p)
    info_all_images = []
    for i in xrange(number_of_images):
        info_all_images.append(postprocess(p[i], gt[i], width_and_height[i], false_positives=False, false_negatives=False))
    return info_all_images


if __name__ == "__main__":
    imdb = get_imdb(imdb_name)
    print(imdb_name)
    imdb.competition_mode(on=True)

    # load net
    net = FasterRCNN(classes=imdb.classes, debug=False)
    network.load_net(trained_model, net)
    print('load model successfully!')

    net.cuda()
    net.eval()
    train_mode = 1  # 1 if on train mode, 0 on test mode
    serialize = 1  # 0 if you just want to load the data, 1 if you want to process it
    main(train=train_mode, serialize=serialize)