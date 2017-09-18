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


repo_of_ground_truth = '/home/revan/VOCdevkit/VOC2007/Annotations'
repo_of_images = '/home/revan/VOCdevkit/VOC2007/JPEGImages'
train_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

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


def main():
    content = get_filenames_of_training_set(train_set)
    print(len(content))
    p, gt, all_width_and_height = create_p_and_gt(content, repo_of_images, classes_dict)
    with open('p_test.pickle', 'wb') as f:
        # Pickle p data structure using the highest protocol available.
        pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
    with open('gt_test.pickle', 'wb') as f:
        # Pickle gt data structure using the highest protocol available.
        pickle.dump(gt, f, pickle.HIGHEST_PROTOCOL)
    with open('width_and_height_test.pickle', 'wb') as f:
        # Pickle gt data structure using the highest protocol available.
        pickle.dump(all_width_and_height, f, pickle.HIGHEST_PROTOCOL)
    with open('i_over_u_test.pickle', 'wb') as f:
        # Pickle gt data structure using the highest protocol available.
        pickle.dump(gt, f, pickle.HIGHEST_PROTOCOL)

    # load the pickle files
    with open('p_test.pickle', 'rb') as f:
        p = pickle.load(f)
    with open('gt_test.pickle', 'rb') as f:
        gt = pickle.load(f)
    with open('width_and_height_test.pickle', 'rb') as f:
        all_width_and_height = pickle.load(f)
    with open('i_over_u_test.pickle', 'rb') as f:
        all_i_over_u = pickle.load(f)

    p_and_gt = postprocess(p, gt, all_width_and_height)
    print("meh")


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
        cls_dets = cls_dets[keep, :]
        if cls_dets.shape[0] != 0:
            for k in xrange(len(cls_dets)):
                all_class_boxes.append(cls_dets[k])

        to_keep.extend(inds[keep])

    all_scores = scores[to_keep]
    class_boxes = np.zeros((len(all_class_boxes), 5))
    for whatever in range(len(all_class_boxes)):
        class_boxes[whatever] = all_class_boxes[whatever][0]
    print(all_scores.shape[0] == class_boxes.shape[0])
    return all_scores, class_boxes


def create_p_and_gt(content, repo_of_images, dict):

    rcnn_output = []
    ground_truth = []
    all_width_and_height = []


    for image_name in content:
        im = cv2.imread(os.path.join(repo_of_images, image_name + ".jpg"))
        xml_file = os.path.join(repo_of_ground_truth, image_name + ".xml")

        all_scores, all_class_boxes = test_net(save_name, net, imdb, im, max_per_image, thresh=thresh, vis=0)

        gt_info, width, height = find_objects_in_files(xml_file, dict)
        ground_truth.append(gt_info)

        if len(all_scores) != 0:
            rcnn_output_individual = np.zeros((len(all_scores), 25))
            rcnn_output_individual[:, :21] = all_scores
            rcnn_output_individual[:, 21:] = all_class_boxes[:, :-1]
            rcnn_output.append(rcnn_output_individual)
            all_width_and_height.append(np.asarray([width, height]))
        else:
            rcnn_output.append(np.asarray([]))  # we add an empty array just to have everything synchronized
            all_width_and_height.append(np.asarray([width, height]))

    return rcnn_output, ground_truth, all_width_and_height


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

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


def postprocess(p, gt, all_width_and_height):
    number_of_images = len(p)

    all_i_over_u = []

    for i in xrange(number_of_images):
        image_p = p[i]
        image_gt = gt[i]
        gt_len = len(image_gt)
        p_len = len(image_p)
        i_over_u = np.zeros((gt_len, p_len))
        for j in xrange(gt_len):
            for k in xrange(p_len):
                _intersection_over_union = bb_intersection_over_union(image_p[k, -4:], image_gt[j, -4:])
                i_over_u[j, k] = _intersection_over_union
        all_i_over_u.append(i_over_u)


    all_p_and_gt_postprocessed = []
    for i in xrange(number_of_images):
        p_and_gt_postprocessed = find_best_matches(all_i_over_u[i], p[i], gt[i])
        p_and_gt_postprocessed.append(all_width_and_height[i])
        all_p_and_gt_postprocessed.append(p_and_gt_postprocessed)
    with open('p_and_gt_test.pickle', 'wb') as f:
        # Pickle p data structure using the highest protocol available.
        pickle.dump(all_p_and_gt_postprocessed, f, pickle.HIGHEST_PROTOCOL)
    return all_p_and_gt_postprocessed


def find_best_matches(numpy_array, p_object, gt_object):
    rows = numpy_array.shape[0]
    columns = numpy_array.shape[1]
    max_number_matches = min(rows, columns)
    p_object_postprocessed = []
    gt_object_postprocessed = []
    rectangles_p = []
    rectangles_gt = []
    p_and_gt_postprocessed = []
    for i in xrange(max_number_matches):
        best_match = unravel_index(numpy_array.argmax(), numpy_array.shape)
        # consider only non-zero intersection over union
        if numpy_array[best_match[0], best_match[1]]:
            # make the entire row and column 0
            numpy_array[best_match[0], :] = 0.
            numpy_array[:, best_match[1]] = 0.
            p_object_postprocessed.append(p_object[best_match[1], :21])
            rectangles_p.append(p_object[best_match[1], 21:])
            gt_object_postprocessed.append(gt_object[best_match[0], :21])
            rectangles_gt.append(gt_object[best_match[0], 21:])

    p_object_postprocessed = np.asarray(p_object_postprocessed)
    gt_object_postprocessed = np.asarray(gt_object_postprocessed)
    rectangles_p = np.asarray(rectangles_p)
    rectangles_gt = np.asarray(rectangles_gt)
    p_and_gt_postprocessed.append(p_object_postprocessed)
    p_and_gt_postprocessed.append(gt_object_postprocessed)
    p_and_gt_postprocessed.append(rectangles_p)
    p_and_gt_postprocessed.append(rectangles_gt)
    return p_and_gt_postprocessed


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
    main()
    print("meh")



