import numpy as np
from input_relab import get_filenames_of_set as files
from numpy import linalg as LA
import pickle


data_structure_train = 'info_all_images_trainval.pickle'
data_structure_test = 'info_all_images_test.pickle'


def main():
    with open('info_all_images_trainval.pickle', 'rb') as f:
        info_all_images = pickle.load(f)
    current_image = info_all_images[0]
    rects = current_image[2]
    width_and_height = current_image[4]
    number_of_rects = len(rects)

    centres = np.zeros((rects.shape[0], 2))
    for i in xrange(number_of_rects):
        centres[i] = normalize_and_centralize_rectangle(rects[i], width_and_height)

    distances = np.zeros((number_of_rects, number_of_rects))
    angles = np.zeros((number_of_rects, number_of_rects))

    for i in xrange(number_of_rects):
        for j in xrange(number_of_rects):
            if i != j:
                distances[i, j], angles[i, j] = find_distance_and_angle(centres[i], centres[j])
    print()




def normalize_and_centralize_rectangle(rect, width_and_height):
    width_and_height.astype(float)
    # normalize the rectangles by dividing with the width and height of the image
    rect[0] /= width_and_height[0]  # we divide the first coordinate of left uppermost point of the rectangle with the width of the image
    rect[2] /= width_and_height[0]  # we divide the first coordinate of right lowermost point of the rectangle with the width of the image
    rect[1] /= width_and_height[1]  # we divide the second coordinate of left uppermost point of the rectangle with the height of the image
    rect[3] /= width_and_height[1]  # we divide the second coordinate of right lowermost point of the rectangle with the height of the image

    # find the centers of the rectangles
    centre = np.zeros(2)  # centre not center, God save her highness the Queen
    centre[0] = (rect[2] + rect[0]) / 2.
    centre[1] = (rect[3] + rect[1]) / 2.

    return centre


def find_distance_and_angle(point1, point2):
    distance = LA.norm(point1 - point2)
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]
    angle = np.arctan2(y, x)
    return distance, angle


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
                distances.append(LA.norm(points[i] - points[j]))
                x = points[j, 0] - points[i, 0]
                y = points[j, 1] - points[i, 1]
                angles.append(np.arctan2(y, x))
    return distances, angles


if __name__ == "__main__":
    main()