#!/usr/bin/env python
import skimage.io
from skimage.transform import rotate
import matplotlib.pyplot as plt
from pycomar.images import show3plt
import numpy as np


x = skimage.io.imread("./srcs/imgs/sample01.jpg")

xs = []

limit = 3
step = 1 

for i in range(-limit, limit + 1, step):
    x_hat = (rotate(x, i) * 255).astype('uint8')
    xs.append(x_hat)

x_blur = np.stack(xs).mean(axis=0).astype('uint8')

skimage.io.imsave("result.jpg", x_blur, quality=100)
