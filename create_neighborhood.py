import numpy as np
import inspect
from input_relab import bb_intersection_over_union

# info all images trainval: list l of lists l_x of ndarrays ar_x
# each element in l represents data related to an image
# each element in l_x contains 5 ndarrays:
# ar_0: softmax of R-CNN
# ar_1: softmax of ground-truth
# ar_2: bboxes of R-CNNs
# ar_3: bboxes of ground-truth
# ar_4: width and height of image (1 x 2)
# ar_5: bboxes for each class
# ar_6: softmax of binary classification [not-background, background]
# ar_7: same of ar_6 for GT


class NeighborhoodGenerator:
    def __init__(self, bins_types, bins_funcs_or_mats):
        self.bins_types = bins_types
        self.bins_funcs_or_mats = bins_funcs_or_mats
        self.D = reduce(lambda x, y: ((len(x) - 1) * (len(y) - 1)), self.bins_types)

    def create_N(self, objs):
        num_objs = len(objs)

        if num_objs < 2:
            raise ValueError("Not enough objects to create a neighborhood")

        N = np.zeros((self.D, num_objs, num_objs))
        bins_matrices = []
        for bins_type, bins_func_or_mat in zip(self.bins_types, self.bins_funcs_or_mats):
            if inspect.isfunction(bins_func_or_mat):
                bins_matrix = bins_func_or_mat(objs)
                for i, (x_0, x_1) in enumerate(zip(bins_type[:-1], bins_type[1:])):
                    bins_matrix[np.logical_and(x_0 <= bins_matrix, bins_matrix <= x_1)] = i
                bins_matrices.append(bins_matrix)
            else:
                bins_matrices.append(bins_func_or_mat)

        ravel_indices = np.array(reversed(range(1, self.bins_types)))

        def aux(tup): return tup[0] ** tup[1]
        for ((i, j), d0), ds_from_1_to_D in zip(np.ndenumerate(self.bins_types[0].astype(int)),
                                                map(lambda x: np.nditer(x).astype(int), self.bins_types[1:])):
            N[sum(list(map(aux, zip(ds_from_1_to_D, ravel_indices))) + [d0]), i, j] += 1

        for i in range(N.shape[0]):
            np.fill_diagonal(N[i], 0.0)
        return N


def generate_IOU_matrix(bboxes):
    """
    Generate iou between each pair of bboxes taken in input.
    :param bboxes:
    :return:
    """
    num_bboxes = len(bboxes)
    iou_matrix = np.zeros((num_bboxes, num_bboxes))

    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes[(i + 1):]):
            iou_matrix[i, i + j] = iou_matrix[i + j, i] = bb_intersection_over_union(bbox1, bbox2)

    return iou_matrix


def generate_scores_matrix(scores):
    num_bboxes = len(scores)

    scores_matrix = np.zeros((num_bboxes, num_bboxes))
    for i, score1 in enumerate(scores):
        for j, score2 in enumerate(scores[(i + 1):]):
            if (score1 < 0.5) and (score2 < 0.5):
                val = 0
            elif ((score1 < 0.5) and (score2 > 0.5)) or ((score2 < 0.5) and (score1 > 0.5)):
                val = 1
            elif (score1 > 0.5) and (score2 > 0.5):
                val = 2
            else:
                raise ValueError("score 1 (" + str(score1) + ") or score 2 (" + str(score2) + ") have invalid values.")
            scores_matrix[i, i + j] = scores_matrix[i + j, i] = val
    return scores_matrix

def create_N(bboxes, d_bins=None, alpha_bins=None):
    if d_bins is None:
        d_bins = np.linspace(0.0, 1.0, 5)
    if alpha_bins is None:
        alpha_bins = np.linspace(0.0, 2.0 * np.pi, 5)

    num_bboxes = len(bboxes)
    if num_bboxes >= 2:
        centres = normalize_and_centralize_bboxes(bboxes)
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

        for i in range(N.shape[0]):
            np.fill_diagonal(N[i], 0.0)
    else:
        raise ValueError("Not enough bounding boxes to create a neighborhood")

    return N


def create_N(bboxes, d_bins=None, alpha_bins=None):
    if d_bins is None:
        d_bins = np.linspace(0.0, 1.0, 5)
    if alpha_bins is None:
        alpha_bins = np.linspace(0.0, 2.0 * np.pi, 5)

    num_bboxes = len(bboxes)
    if num_bboxes >= 2:
        centres = normalize_and_centralize_bboxes(bboxes)
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

        for i in range(N.shape[0]):
            np.fill_diagonal(N[i], 0.0)
    else:
        raise ValueError("Not enough bounding boxes to create a neighborhood")

    return N


def merge_angle_neighbourhood(N, d_bins):
    # NB: for now it works only with no distance bins
    new_N = np.zeros((3 * len(d_bins) - 1, N.shape[1], N.shape[1]))

    for i in xrange(len(d_bins) - 1):
        new_N[0 + (i * 3)] = N[0 + (i * 5)] + N[2 + (i * 5)] + N[4 + (i * 5)]
        new_N[1 + (i * 3)] = N[1 + (i * 5)]
        new_N[2 + (i * 3)] = N[3 + (i * 5)]
    return new_N


def normalize_and_centralize_bboxes(bboxes):
    # normalize the rectangles by dividing with the width and height of the image
    bboxes_transformed = np.zeros((bboxes.shape[0], 4))
    bboxes_transformed[:, [0, 2]] = bboxes[:, [0, 2]]
    bboxes_transformed[:, [1, 3]] = bboxes[:, [1, 3]]

    # find the centers of the rectangles
    centres = np.zeros((bboxes_transformed.shape[0], 2))  # centres not centers, God save her highness the Queen
    centres[:, 0] = (bboxes_transformed[:, 0] + bboxes_transformed[:, 2]) / 2.0
    centres[:, 1] = (bboxes_transformed[:, 1] + bboxes_transformed[:, 3]) / 2.0

    return centres


def find_centre(bbox):
    # find the centers of the rectangles
    centre = np.zeros(2)  # centres not centers, God save her highness the Queen
    centre[0] = (bbox[0] + bbox[2]) / 2.0
    centre[1] = (bbox[1] + bbox[3]) / 2.0

    return centre


def find_dists_and_angles(points):
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

