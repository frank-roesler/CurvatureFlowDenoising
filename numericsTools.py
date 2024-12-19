import numpy as np
from imageProcessing import *


def D0x(img: np.array) -> np.array:
    imgPlus = np.roll(img, 1, axis=1)
    imgMinus = np.roll(img, -1, axis=1)
    return removeBoundary((imgMinus - imgPlus) / 2)


def D0y(img: np.array) -> np.array:
    imgPlus = np.roll(img, 1, axis=0)
    imgMinus = np.roll(img, -1, axis=0)
    return removeBoundary((imgPlus - imgMinus) / 2)


imgSize = 1024
T1 = getTestImage(imgSize, imgSize)

D0yT1 = D0y(T1)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im1 = ax[0].imshow(T1)
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(D0yT1)
fig.colorbar(im2, ax=ax[1])

plt.show()
