
import matplotlib.pyplot as plt
import random
import cv2
import math
import numpy as np
from tqdm import tqdm


def ComputeGraph(img, b, g, r):
    height, width, _ = img.shape  # 133,185
    edges_size = height*width*4
    # edges = np.zeros(shape=(edges_size, 3), dtype=object)
    edges = []
    num = 0
    for i in range(height):
        for j in range(width):
            if i < height-1:
                # edges[num, 0] = (i, j)
                # edges[num, 1] = (i+1, j)
                # edges[num, 2] = GetDifference(i, j, i+1, j, b, g, r)
                edges.append([(i, j), (i+1, j),
                              GetDifference(i, j, i+1, j, b, g, r)])
                num += 1
            if j < width-1:
                # edges[num, 0] = (i, j)
                # edges[num, 1] = (i, j+1)
                # edges[num, 2] = GetDifference(i, j, i, j+1, b, g, r)
                edges.append([(i, j), (i, j+1),
                              GetDifference(i, j, i, j+1, b, g, r)])
                # print(edges[num])
                num += 1
            if i < height-1 and j < width-2:
                # edges[num, 0] = (i, j)
                # edges[num, 1] = (i+1, j+1)
                # edges[num, 2] = GetDifference(i, j, i+1, j+1, b, g, r)
                edges.append([(i, j), (i+1, j+1),
                              GetDifference(i, j, i+1, j+1, b, g, r)])
                num += 1
            if i < height-1 and j > 0:
                # edges[num, 0] = (i, j)
                # edges[num, 1] = (i+1, j-1)
                # edges[num, 2] = GetDifference(i, j, i+1, j-1, b, g, r)
                edges.append([(i, j), (i+1, j-1),
                              GetDifference(i, j, i+1, j-1, b, g, r)])
                num += 1
    return edges


def MergeToComponent(edges, c):
    edges = sorted(edges, key=lambda edges: edges[2])
    vertice_set = [[] * 100 for i in range(100)]
    for i in tqdm(range(len(edges))):
        if edges[i][2] < c:
            for j in range(len(vertice_set)):
                if edges[i][0] in vertice_set[j]:
                    vertice_set[j].append(edges[i][1])
                else:
                    vertice_set[j].append(edges[i][0])
                    vertice_set[j].append(edges[i][1])

    return vertice_set


def GetDifference(x1, y1, x2, y2, B, G, R):
    pixel_diff = math.sqrt((B[x1][y1]-B[x2][y2])**2 +
                           (G[x1][y1]-G[x2][y2])**2+(R[x1][y1]-R[x2][y2])**2)
    return pixel_diff


def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb


if __name__ == '__main__':
    img = cv2.imread("./test.jpg")
    (b, g, r) = cv2.split(img)
    height, width, _ = img.shape
    edges_size = height*width*4
    edges = ComputeGraph(img, b, g, r)
    # edges = [x for x in edges if x != 0 0 0]]
    vertice_set = MergeToComponent(edges, 1)
    print(vertice_set)
    colors = np.zeros(shape=(height * width, 3))
    output = np.zeros(shape=(height, width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()

    for i in vertice_set:
        output[i[0], i[1], :] = colors[i, :]
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    a.set_title('Original Image')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(output)
    a.set_title('Segmented Image')
    plt.show()

    # for i in edges:
    #     print(i)
