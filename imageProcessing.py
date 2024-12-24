import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from parameterValues import myEpsilon


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
    return np.array(img) / 255.0


def removeBoundary(imgArray: np.array) -> np.array:
    return imgArray[1:-1, 1:-1]


def getTestImage(height: int, width: int) -> np.array:
    result = np.zeros((height, width))
    barWidth = width // 16
    result[height // 4 : -height // 4, width // 2 - barWidth : width // 2 + barWidth] = 1
    result[height // 4 - barWidth : height // 4 + barWidth, width // 8 : -width // 8] = 1

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


def getTestImageGrayscale(height: int, width: int) -> np.array:
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    result = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            result[y, x] = 1 - (distance / max_distance)

    return result


def circularFilter(stencilSize: int) -> np.array:
    if stencilSize == 0:
        return np.ones((2, 2)) / 4
    radius = stencilSize
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2 * stencilSize + 1, 2 * stencilSize + 1))
    kernel[mask] = 1
    return kernel / np.sum(kernel)


def twoPointFilter(unitVector: np.array, stencilSize: int) -> np.array:
    x = np.linspace(-1, 1, 2 * stencilSize + 1)
    y = np.linspace(1, -1, 2 * stencilSize + 1)
    kernel = np.zeros((2 * stencilSize + 1, 2 * stencilSize + 1))
    indX = np.argmin(np.abs(x - unitVector[0]))
    indY = np.argmin(np.abs(y + unitVector[1]))
    kernel[indX, indY] = 1
    kernel[2 * stencilSize - indX, 2 * stencilSize - indY] = 1
    return kernel


def applyTwoPointFilters(img: np.array, DxImg: np.array, DyImg: np.array, stencilSize: int) -> np.array:
    """Applies tangenmtial two-point averages to image."""
    h, w = img.shape
    imgPadded = np.pad(img, pad_width=stencilSize, mode="reflect")
    imgGradient = np.stack([DxImg, DyImg], axis=-1)
    gradNorms = np.linalg.norm(imgGradient, axis=-1)
    imgGradient[gradNorms <= myEpsilon, :] = 0
    imgGradient[gradNorms > myEpsilon] = (
        imgGradient[gradNorms > myEpsilon] / gradNorms[gradNorms > myEpsilon, np.newaxis]
    )
    x = np.linspace(-1, 1, 2 * stencilSize + 1)
    y = np.linspace(1, -1, 2 * stencilSize + 1)
    result = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            pixelTile = getTile(imgPadded, i, j, stencilSize)
            grad = imgGradient[i, j]
            iX = np.argmin(np.abs(x - grad[0]))
            iY = np.argmin(np.abs(y + grad[1]))
            result[i, j] = (pixelTile[iX, iY] + pixelTile[2 * stencilSize - iX, 2 * stencilSize - iY]) / 2
            # filter = np.zeros(pixelTile.shape)
            # for factor in np.linspace(-1, 1, 2 * stencilSize):
            #     grad = factor * imgGradient[i, j]
            #     iX = np.argmin(np.abs(x - grad[0]))
            #     iY = np.argmin(np.abs(y + grad[1]))
            #     filter[iX, iY] = 1
            # result[i, j] = np.sum(pixelTile * filter) / np.sum(filter)
    return result


def getTile(paddedImage: np.array, i: int, j: int, stencilSize: int) -> np.array:
    if stencilSize == 0:
        return paddedImage[i : i + 2, j : j + 2]
    return paddedImage[i : i + 2 * stencilSize + 1, j : j + 2 * stencilSize + 1]


def localAverage(img: np.array, stencilSize: int) -> np.array:
    kernel = circularFilter(stencilSize)
    return convolve(img, kernel, mode="reflect")
