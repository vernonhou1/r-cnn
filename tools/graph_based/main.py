from image_seg import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():

    sigma = 0.5
    k = 1000
    min_size = 50

    img = cv2.imread("./test.jpg")
    float_img = np.asarray(img, dtype=float)
    gaussian_img = cv2.GaussianBlur(float_img, (5, 5), sigma)
    height, width, channel = img.shape
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)

    edges, num_vert, num = GetEdges(smooth_img, width, height)
    disjoint_set = GetSegment(num_vert, edges, num, k)
    disjoint_set = ProcessSmallCom(num, disjoint_set, edges, min_size)
    output = GenerateImage(disjoint_set, width, height)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    a.set_title('Original Image')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(output)
    a.set_title('Segmented Image')
    plt.show()


if __name__ == '__main__':
    main()
