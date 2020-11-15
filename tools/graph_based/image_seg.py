import numpy as np
import cv2
import math
from disjoint_set import *
import random as rand


def GetDiff(img, x1, y1, x2, y2):
    r = (img[0][y1, x1] - img[0][y2, x2]) ** 2
    g = (img[1][y1, x1] - img[1][y2, x2]) ** 2
    b = (img[2][y1, x1] - img[2][y2, x2]) ** 2
    return math.sqrt(r + g + b)


def GetThreshold(k, size):
    return (k/size)


def GetEdges(img, width, height):
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for j in range(height):
        for i in range(width):
            if i < width - 1:
                edges[num, 0] = int(j*width + i)
                edges[num, 1] = int(j*width+i+1)
                edges[num, 2] = GetDiff(img, i, j, i+1, j)
                num += 1
            if j < height-1:
                edges[num, 0] = int(j*width + i)
                edges[num, 1] = int((j+1)*width+i)
                edges[num, 2] = GetDiff(img, i, j, i, j+1)
                num += 1
            if (i < width - 1) and (j < height - 1):
                edges[num, 0] = int(j * width + i)
                edges[num, 1] = int((j + 1) * width + (i + 1))
                edges[num, 2] = GetDiff(img, i, j, i+1, j+1)
                num += 1
            if (i < width - 1) and (j > 0):
                edges[num, 0] = int(j * width + i)
                edges[num, 1] = int((j - 1) * width + (i + 1))
                edges[num, 2] = GetDiff(img, i, j, i+1, j-1)
                num += 1
    return edges, height*width, num


def GetSegment(num_vert, edges, num, k):
    edges[0:num, :] = edges[edges[0:num, 2].argsort()]
    disjoint_set = universe(num_vert)
    threshold = np.zeros(shape=num_vert, dtype=float)
    for i in range(num_vert):
        threshold[i] = GetThreshold(1, k)
    for i in range(num):
        pedge = edges[i, :]

    # components connected by this edge
        a = disjoint_set.find(pedge[0])
        b = disjoint_set.find(pedge[1])
        if a != b:
            if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                disjoint_set.join(a, b)
                a = disjoint_set.find(a)
                threshold[a] = pedge[2] + \
                    GetThreshold(disjoint_set.size(a), k)

    return disjoint_set


def ProcessSmallCom(num, disjoint_set, edges, min_size):
    for i in range(num):
        a = disjoint_set.find(edges[i, 0])
        b = disjoint_set.find(edges[i, 1])
        if (a != b) and ((disjoint_set.size(a) < min_size) or (disjoint_set.size(b) < min_size)):
            disjoint_set.join(a, b)

    return disjoint_set


def GenerateImage(disjoint_set, width, height):
    def random_color(): return (int(rand.random() * 255),
                                int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(width * height)]
    save_img = np.zeros((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            color_idx = disjoint_set.find(y * width + x)
            save_img[y, x] = color[color_idx]

    return save_img
