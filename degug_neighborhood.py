import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import warnings
import cv2
import os
from input_relab import get_filenames_of_set as files


repo_of_images = '/home/revan/VOCdevkit/VOC2007/JPEGImages'
data_structure_name = 'info_all_images_trainval.pickle'
train_set = '/home/revan/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'


def main():
    thetas = np.array([[0, np.pi], [0, np.pi], [-np.pi, 0], [-np.pi, 0]])
    dists = np.array([[0, 0.2], [0.2, 0.5], [0.5, 0.7], [0.7, 1.00001]])

    content = files(train_set)

    Ps, Ps_L, Ns, R, images_ids, bboxes, img_sizes = read_rcnn_train_data(data_structure_name, thetas, dists)

    j = 0
    for i, im in enumerate(content):
        if images_ids[i]:
            image = cv2.imread(os.path.join(repo_of_images, im + ".jpg"))
            box = bboxes[i]
            print(Ns[j])
            number_of_objects = len(Ps[j])
            for k in xrange(number_of_objects):
                cv2.rectangle(image, (int(box[k, 0]), int(box[k, 1])), (int(box[k, 2]), int(box[k, 3])),
                              box[k], thickness=1, lineType=4)
                cv2.imshow('image', image)
                cv2.waitKey(0)
                j += 1


def get_bbox_neighbours(p, thetas, dists, dtype=float):
    # return a tensor of size dxnxn,
    # thetas and dists are both of size kx2 intervals

    if p.shape[0] == 1:
        # if only one bbox has been detected
        return np.zeros((thetas.shape[0], 1, 1), dtype=dtype)

    centers = np.zeros((p.shape[0], 2), dtype=dtype)
    centers[:, 0] = (p[:, 2] + p[:, 0]) / 2.0
    centers[:, 1] = (p[:, 3] + p[:, 1]) / 2.0

    all_dists = squareform(pdist(centers))

    all_angles = np.zeros((p.shape[0], p.shape[0]))
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            # move the cartesian center to the i-th center
            pt = centers[j, :] - centers[i, :]
            # compute the angle between the new center and the center of the j-th bbox
            all_angles[i, j] = np.arctan2(pt[1], pt[0])

    N = np.zeros((thetas.shape[0], p.shape[0], p.shape[0]), dtype=dtype)

    for k, (theta, dist) in enumerate(zip(thetas, dists)):
        if k < thetas.shape[0] - 1:
            D = (all_dists >= dist[0]) * (all_dists < dist[1])
            A = (all_angles >= theta[0]) * (all_angles < theta[1])
        else:
            # includes also the extrema in the last bin
            # TODO: to be corrected because the last bin is not the last in the sequence!
            D = (all_dists >= dist[0]) * (all_dists <= dist[1])
            A = (all_angles >= theta[0]) * (all_angles <= theta[1])

        N[k, :, :] = np.array(D * A * np.logical_not(np.eye(p.shape[0])), dtype=dtype)

    return N


def read_rcnn_train_data(fn="rcnn_data/p_and_gt.pickle", thetas=None, dists=None, dtype=float):
    file = open(fn, 'rb')
    ps = pickle.load(file)

    Ns = []
    Ps = []
    Ps_L = []
    images_ids = np.zeros((len(ps), 1), dtype=dtype)

    bboxes = []
    img_sizes = []

    for i, p in enumerate(ps):
        print(i)
        if p[0].shape[0] > 1:
            if not (thetas is None) and not (dists is None):
                Ns.append(get_bbox_neighbours(p[2] / np.hstack((p[-1], p[-1]))[np.newaxis, :], thetas, dists, dtype))
            else:
                Ns.append(
                    np.ones((1, p[0].shape[0], p[0].shape[0]), dtype=dtype) - np.eye(p[0].shape[0], dtype=dtype))

            Ps.append(p[0].astype(dtype))
            Ps_L.append(p[1].astype(dtype))
            images_ids[i] = 1.0
        else:
            warnings.warn("The " + str(i) + "-th p is empty or made by just one bbox.")

        bboxes.append(p[3])
        img_sizes.append(p[4])

    R = np.random.rand(Ns[0].shape[0], Ps[0].shape[1], Ps[0].shape[1]).astype(dtype)

    return Ps, Ps_L, Ns, R, images_ids, bboxes, img_sizes


def read_rcnn_test_data(fn="rcnn_data/p_and_gt_test.pickle", thetas=None, dists=None, dtype=float):
    Ps, Ps_L, Ns, R, image_ids, bboxes, img_sizes = read_rcnn_train_data(fn, thetas, dists, dtype)
    return Ps, Ps_L, Ns, image_ids, bboxes, img_sizes


def convert_to_pascalvoc(ps, valid_images, bboxes, img_sizes, without_background=0):
    all_boxes = [[np.empty((0, 5)) for _ in range(valid_images.shape[0])] for _ in range(ps[0].shape[1]+without_background)] # labels

    c = 0
    for i in range(valid_images.shape[0]):  # images
        if valid_images[i] == 1:
            cls = np.argmax(ps[c], axis=1)
            for j in range(cls.shape[0]):
                original_bb = [bboxes[i][j][0], bboxes[i][j][1],
                               bboxes[i][j][2], bboxes[i][j][3]]
                all_boxes[cls[j]+without_background][i] = np.vstack((all_boxes[cls[j]][i], np.hstack((original_bb, np.max(ps[c][j])))))
            c += 1

    return all_boxes

if __name__ == '__main__':
    main()
