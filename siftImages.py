import cv2
import numpy as np
import sys

def resize_img(img):
    """
    downsizes an image if it has a height larger than 600
    and/or a width larger than 480 while keeping aspect ratio.

    :param img: image to be resized
    :return: image with new or original size
    """

    height, width = img.shape[:2]

    if height > 600:
        new_height = 600
        ratio = float(new_height / height)
        height = new_height
        width = int(width * ratio)

    if width > 480:
        new_width = 480
        ratio = float(new_width / width)
        width = new_width
        height = int(height * ratio)

    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized_img


def prepare_image(img):
    """
    converts image to YCrCb and extracts y component.

    :param img: image to be converted
    :return: y component of image
    """

    # convert to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # get Y component to form grayscale image
    # y, cb, cr = cv2.split(ycrcb_img)
    y_component = ycrcb_img[:, :, 0]

    return y_component


def find_keypoints(img):
    """
    finds keypoints and descriptors using SURF detector.

    :param img: image from which to detect keypoints
    :return: keypoints and descriptors
    """

    # create SURF detector
    sift = cv2.SIFT_create(nfeatures=800) 

    # get keypoints and descriptors
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def draw_keypoints(img, keypoints):
    """
    draws keypoints on an image.

    :param img: image on which to draw the keypoints
    :param kp: list of keypoints
    :return: image highlighted with keypoints
    """

    # for each keypoint the circle around keypoint with keypoint size and orientation will be drawn
    new_img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for keypoint in keypoints:
        # get coordinates for each keypoint
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])

        # for each keypoint draw a cross
        new_img = cv2.drawMarker(new_img, (x, y), (255, 102, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

    return new_img


def combine_images(img1, img2):
    """
    combines two images into one for display.

    :param img1: first image
    :param img2: second image
    :return: combined image
    """

    return np.concatenate((img1, img2), axis=1)


def show_image(img):
    """
    displays image with specified title.

    :param img: image to be displayed
    """

    cv2.imshow("Original Image and Image with Keypoints", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_keypoints_program(filename):
    """
    extracts keypoints (and descriptors) from the Y component of an image
    and displays the original image and highlighted image.

    :param filename: filename of image to be analysed
    """

    original_img = cv2.imread(filename)
    original_img = resize_img(original_img)
    prepared_img = prepare_image(original_img)
    keypoints, descriptors = find_keypoints(prepared_img)
    marked_img = draw_keypoints(original_img, keypoints)
    combined = combine_images(original_img, marked_img)
    show_image(combined)
    print("# of keypoints in {0} is {1}".format(filename, len(keypoints)))


def prepare_descriptors(des_list):
    """
    prepares descriptors for kmeans by combining
    descriptors for all images into one vertical numpy
    array of float32 type.

    :param des_list: a list where each element is a list of
                     descriptors for different images
    :return: vertical numpy array of all descriptors as float32
    """
    
    descriptors = []

    # stack all descriptors vertically in numpy array
    for descriptor in des_list:
        if len(descriptors) == 0:
            descriptors = descriptor
        else:
            descriptors = np.vstack((descriptors, descriptor))

    # convert integers to float
    descriptors_float = np.float32(descriptors)
    
    return descriptors_float


def k_means(descriptors, percentage):
    """
    clusters descriptors from all images into K-clusters
    using K-means algorithm, where K is a percentage of the
    total number of keypoints.

    :param descriptors: list of all descriptors stacked vertically
    :param percentage: a specified percentage
    :return: list of labels corresponding to a cluster (visual word)
    """

    # specify K as percentage of total number of keypoints
    k = int(percentage * len(descriptors) / 100)

    # whenever 10 iterations of algorithm is ran, or an accuracy of
    # epsilon = 1.0 is reached, stop the algorithm and return the answer
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # apply kmeans()
    compactness, labels, centers = cv2.kmeans(descriptors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return labels


def construct_histograms(keypoints, labels):
    """
    constructs a histogram for each image of the occurrence of
    the labels corresponding to visual words.

    :param keypoints: a list of lists of keypoints where each individual
                      list contains the keypoints of one image
    :param labels: list of labels corresponding to a cluster (visual word)
    :return: list of normalised histograms
    """

    histograms = []
    start = 0
    num_clusters = len(np.unique(labels))

    # for each image, construct histogram of the occurrence of visual words
    for img_keypoints in keypoints:
        end = start + len(img_keypoints) + 1

        # get image labels
        image_labels = labels[start:end]

        histogram, _ = np.histogram(image_labels, bins=num_clusters, range=(0, num_clusters), density=True)
        histograms.append(histogram)

        # go to next image labels
        start += len(img_keypoints) + 1

    return histograms


def measure_dissimilarity(histograms):
    """
    calculates the chi square distance between the histograms.

    :param histograms: list of histograms
    :return: matrix of distances of all histogram pairs
    """

    dissimilarity_matrix = np.zeros((len(histograms), len(histograms)))

    for i in range(len(histograms)):
        hist1 = np.float32(histograms[i])

        for j in range(len(histograms)):
            hist2 = np.float32(histograms[j])

            # opencv's regular chi square distance for comparing histograms does not give symmetric values
            # therefore, alternative chi square distance is used to get symmetric values
            # we divide by 4 because opencv's alternative chi square is factor 4
            dissimilarity_matrix[i,j] = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CHISQR_ALT) / 4

    return dissimilarity_matrix


def print_keypoints(images, keypoints):
    """
    prints the number of keypoints found in each image.

    :param images: list of image filenames
    :param keypoints: list of lists of keypoints for each image
    """

    for i, image in enumerate(images):
        print("# of keypoints in {0} is {1}".format(image, len(keypoints[i])))

    print("# of keypoints in total is {0}".format(sum([len(kp) for kp in keypoints])))
    print()


def print_dissimilarities(images, keypoints, dissimilarities, percentage):
    """
    prints a dissimilarity matrix of all the chi square distances between images.

    :param images: list of image filenames
    :param keypoints: list of lists of keypoints in each image
    :param dissimilarities: matrix of distances of all histogram pairs
    :param percentage: specified percentage
    """

    total_kp = sum([len(kp) for kp in keypoints])
    k = int(percentage * total_kp / 100)

    matrix_string = "K = {0}% * {1} = {2}\n\n".format(percentage, total_kp, k)
    matrix_string += "Dissimilarity Matrix\n\n"

    first_line = "{0:<10}".format("")
    remaining_lines = ""

    for i, image in enumerate(images):
        first_line += "{0:>15}".format(image)

        if i == len(images)-1:
            first_line += "\n"

        remaining_lines += "{0:<10}".format(image)

        for j in range(len(images)):
            distance = round(dissimilarities[i,j], 2)

            remaining_lines += "{0:>15}".format(distance)

            if j == len(images)-1:
                remaining_lines += "\n"

    matrix_string += first_line
    matrix_string += remaining_lines

    print(matrix_string)


def run_dissimilarity_program(filenames):
    """
    extracts keypoints and descriptors from the Y components of images
    in a list and clusters descriptors into K-clusters representing visual
    words. constructs histograms for each image of the occurrence of visual
    words and calculates the chi square distances for a dissimilarity matrix.

    :param filenames: list of filenames of images to be analysed
    """
    
    percentages = [5, 10, 20]
    des_list = []
    keypoint_list = []

    # extract all keypoints and descriptors from the Y component of all scaled images
    for file in filenames:
        original_img = cv2.imread(file)
        original_img = resize_img(original_img)
        prepared_img = prepare_image(original_img)
        keypoints, descriptors = find_keypoints(prepared_img)
        des_list.append(descriptors)
        keypoint_list.append(keypoints)

    print_keypoints(filenames, keypoint_list)

    # cluster descriptors from all images into K-clusters for all percentages (5%, 10%, 20%)
    # construct histograms, calculate distances and print result
    for p in percentages:
        descriptors = prepare_descriptors(des_list)
        labels = k_means(descriptors, p)
        histograms = construct_histograms(keypoint_list, labels)
        dissimilarities = measure_dissimilarity(histograms)
        print_dissimilarities(filenames, keypoint_list, dissimilarities, p)


def main():
    """
    runs either keypoint program or dissimilarity program based on command arguments.
    assumes arguments will either be one image filename or a list of image filenames.
    """

    args = sys.argv
    image_files = args[1:]

    if len(image_files) > 1:
        run_dissimilarity_program(image_files)

    else:
        run_keypoints_program(image_files[0])


if __name__ == '__main__':
    main()
