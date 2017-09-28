import numpy as np
import pickle

# info all images trainval: list l of lists l_x of ndarrays ar_x
# each element in l represents data related to an image
# each element in l_x contains 5 ndarrays:
# ar_0: softmax of R-CNN
# ar_1: softmax of ground-truth
# ar_2: bboxes of R-CNNs
# ar_3: bboxes of ground-truth
# ar_4: width and height of image (1 x 2)


def create_Ns(fname='info_all_images_trainval.pickle', d_bins=None, alpha_bins=None):
    if d_bins is None:
        d_bins = np.linspace(0.0, 1.0, 5)
    if alpha_bins is None:
        alpha_bins = np.linspace(0.0, 2.0 * np.pi, 5)

    with open(fname, 'rb') as f:
        images_data = pickle.load(f)

    Ns = []
    for image_data in images_data:
        bboxes = image_data[2]
        img_sz = image_data[4]
        num_bboxes = len(bboxes)

        # centres = np.zeros((bboxes.shape[0], 2))
        # for i, bbox in enumerate(bboxes):
        #     centres[i] = normalize_and_centralize_bbox(bbox, img_sz)

        centres = normalize_and_centralize_bboxes(bboxes, img_sz)

        dists = np.zeros((num_bboxes, num_bboxes))
        angles = np.zeros((num_bboxes, num_bboxes))

        # for i in xrange(num_bboxes):
        #     for j in xrange(num_bboxes):
        #         dists[i, j], angles[i, j] = find_distance_and_angle(centres)
        # np.fill_diagonal(dists, 0.0)

        dists, angles = find_dists_and_angles(centres)

        D = (len(d_bins) - 1) * (len(alpha_bins) - 1)

        N = np.zeros((D, num_bboxes, num_bboxes))
        for i, (x_0, x_1) in enumerate(zip(d_bins[:-1], d_bins[1:])):
            dists[np.logical_and(x_0 <= dists, dists <= x_1)] = i

        angles += np.pi
        for i, (x_0, x_1) in enumerate(zip(alpha_bins[:-1], alpha_bins[1:])):
            angles[np.logical_and(x_0 <= angles, angles <= x_1)] = i

        for ((i, j), d), alpha in zip(np.ndenumerate(dists.astype(int)), np.nditer(angles.astype(int))):
            N[d * (len(d_bins) - 1) + alpha, i, j] += 1

        for i in xrange(N.shape[0]):
            np.fill_diagonal(N[i], 0.0)

        Ns.append(N)
    return Ns


def normalize_and_centralize_bboxes(bboxes, img_sz):
    img_sz = img_sz.astype(float)

    # normalize the rectangles by dividing with the width and height of the image
    bboxes[:, [0, 2]] /= img_sz[0]
    bboxes[:, [1, 3]] /= img_sz[1]

    # find the centers of the rectangles
    centres = np.zeros((bboxes.shape[0], 2))  # centres not centers, God save her highness the Queen
    centres[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    centres[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

    return centres


def find_dists_and_angles(points):
    # distance = np.linalg.norm(points1 - point2)
    # x = point2[0] - point1[0]
    # y = point2[1] - point1[1]
    # angle = np.arctan2(y, x)

    dists = np.zeros((points.shape[0], points.shape[0]))
    angles = np.zeros((points.shape[0], points.shape[0]))
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            sub = p1 - p2
            dists[i, j] = np.linalg.norm(sub)
            angles[i, j] = np.arctan2(sub[1], sub[0])

    np.fill_diagonal(angles, 0.0)
    return dists, angles


def find_distances_and_angles(points):
    """
    This function finds the distances and angles between centers of the bounding boxes
    The width and height of the bounding boxes has been normalized by dividing with the width and height of the image
    :param points: a 2d array representing the centers of bounding boxes
    :return: the following data structures have n by n size, where n is the number of points
           distances - the relative distances between all centers of the bounding boxes
           angles - the relative angles between the centers of the all bounding boxes
    """
    distances = []
    angles = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if i != j:
                distances.append(np.linalg.norm(points[i] - points[j]))
                x = points[j, 0] - points[i, 0]
                y = points[j, 1] - points[i, 1]
                angles.append(np.arctan2(y, x))
    return distances, angles
