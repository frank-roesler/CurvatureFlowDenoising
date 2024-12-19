import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def isGrayScale(img: Image) -> bool:
    img = img.convert("RGB")
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True


def convertToGrayScale(imgPath: str) -> Image:
    img = Image.open(imgPath)
    if not isGrayScale(img):
        img = img.convert("L")
    return img


def addReflectionPadding(imgArray: np.array) -> np.array:
    h, w = imgArray.shape
    firstRow = imgArray[0, :]
    lastRow = imgArray[h - 1, :]
    firstColumn = imgArray[:, 0]
    lastColumn = imgArray[:, w - 1]
    result = np.zeros((h + 2, w + 2))
    result[1 : h + 1, 1 : w + 1] = imgArray
    result[0, 1 : w + 1] = firstRow
    result[h + 1, 1 : w + 1] = lastRow
    result[1 : h + 1, 0] = firstColumn
    result[1 : h + 1, w + 1] = lastColumn
    result[0, 0] = imgArray[0, 0]
    result[0, w + 1] = imgArray[0, w - 1]
    result[h + 1, 0] = imgArray[h - 1, 0]
    result[h + 1, w + 1] = imgArray[h - 1, w - 1]
    return result


def removeBoundary(imgArray: np.array) -> np.array:
    return imgArray[1:-1, 1:-1]


def getTestImage(height: int, width: int) -> np.array:
    result = np.zeros((height, width))
    barWidth = width // 16
    result[
        height // 4 : -height // 4, width // 2 - barWidth : width // 2 + barWidth
    ] = 1
    result[
        height // 4 - barWidth : height // 4 + barWidth, width // 8 : -width // 8
    ] = 1

    # Add filled discs
    radii = [2**k for k in range(int(np.log2(height)))]
    center_x = width - width // 8
    center_y = 2 * height // 5
    for i, radius in enumerate(radii):
        if center_y > height:
            break
        y, x = np.ogrid[-center_y : height - center_y, -center_x : width - center_x]
        mask = x * x + y * y <= radius * radius
        result[mask] = 1
        shift = 2 * (radius + radii[i - 1]) if i > 0 else 2 * radius
        center_y += shift + height // 16
    return result


# img = np.array(convertToGrayScale("IMG_1638_coarse.JPG"))
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img)
# ax[1].imshow(addReflectionPadding(img))
# plt.show()

# fig, ax = plt.subplots(1, 4, figsize=(16, 4))
# ax[0].imshow(getTestImage(128, 256))
# ax[1].imshow(getTestImage(256, 128))
# ax[2].imshow(getTestImage(128, 128))
# ax[3].imshow(getTestImage(64, 64))
# plt.show()
