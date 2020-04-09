import cv2
import numpy as np

def sample(imgs, split=None ,figure_size=(2, 3), img_dim=96, path=None, num=0):
    h, w = figure_size
    if split is None:
        split = range(len(imgs)+1)
    figure = np.zeros((img_dim*h, img_dim*w, 3))
    for i in range(h):
        for j in range(w):
            idx = i*w+j
            if idx >= len(split)-1: break
            digit = imgs[ split[idx] : split[idx+1] ]
            if len(digit) == 1:
                for k in range(3):
                    figure[i*img_dim: (i+1)*img_dim,
                           j*img_dim: (j+1)*img_dim, k] = digit
            elif len(digit) == 3:
                for k in range(3):
                    figure[i*img_dim: (i+1)*img_dim,
                           j*img_dim: (j+1)*img_dim, k] = digit[2-k]
    if path is None:
        cv2.imshow('Figure%d'%num, figure)
        cv2.waitKey()
    else:
        figure *= 255
        print(">> Saving Image at {}".format(path))
        cv2.imwrite(path, figure)