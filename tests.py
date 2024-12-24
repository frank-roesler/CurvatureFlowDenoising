from imageProcessing import *
from numericsTools import *
import matplotlib.pyplot as plt
from parameterValues import *


# ---------------------------------------------------------
# region <TEST PADDING>
# ---------------------------------------------------------
# img = np.array(convertToGrayScale("IMG_1638_coarse.JPG"))
# img1 = addReflectionPadding(img)
# print(img.shape)
# print(img1.shape)
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].imshow(img)
# ax[1].imshow(img1)
# endregion

# ---------------------------------------------------------
# region <TEST LOCAL AVERAGE>
# ---------------------------------------------------------
# img = np.array(convertToGrayScale("IMG_1638_coarse.JPG"))
# img1 = localAverage(img, 1)
# img2 = localAverage(img, 2)
# img3 = localAverage(img, 3)
# print(img.shape)
# print(img1.shape)
# print(img2.shape)
# print(img3.shape)
# fig, ax = plt.subplots(2, 2, figsize=(10, 8))
# ax[0, 0].imshow(img)
# ax[0, 1].imshow(img1)
# ax[1, 0].imshow(img2)
# ax[1, 1].imshow(img3)
# endregion

# ---------------------------------------------------------
# region <TEST GET TEST IMAGE>
# ---------------------------------------------------------
# fig, ax = plt.subplots(1, 4, figsize=(16, 4))
# ax[0].imshow(getTestImage(128, 256))
# ax[1].imshow(getTestImage(256, 128))
# ax[2].imshow(getTestImage(128, 128))
# ax[3].imshow(getTestImage(64, 64))


# imgSize = 64
# T1 = getTestImage(imgSize, imgSize)
# nablaPlus, nablaMinus, DxxImg, DyyImg, DxyImg = computeSignedDerivatives(T1)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# im1 = ax[0].imshow(T1)
# fig.colorbar(im1, ax=ax[0])
# im2 = ax[1].imshow(nablaPlus + nablaMinus)
# fig.colorbar(im2, ax=ax[1])

# plt.show()
# endregion

# ---------------------------------------------------------
# region <TEST TWO POINT FILTER>
# ---------------------------------------------------------
# stencilSize = 3
# disk = circularFilter(stencilSize) / np.max(circularFilter(stencilSize))
# phi = 0.99 * np.pi
# unitVector = np.array([np.cos(phi), np.sin(phi)])
# unitVector = unitVector / np.linalg.norm(unitVector)
# pointFilter = twoPointFilter(unitVector, stencilSize)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(pointFilter)
# ax[1].imshow(disk)
# plt.show()
# endregion


# ---------------------------------------------------------
# region <TEST TWO POINT FILTERS>
# ---------------------------------------------------------
def twoPointFiltersTest(img: np.array, DxImg: np.array, DyImg: np.array, stencilSize: int) -> np.array:
    """auxiliary for testing whether two point filters are generated correctly."""
    h, w = img.shape
    imgPadded = np.pad(img, pad_width=stencilSize, mode=imagePadding)
    imgGradient = np.stack([DxImg, DyImg], axis=-1)
    gradNorms = np.linalg.norm(imgGradient, axis=-1)
    imgGradient[gradNorms <= myEpsilon, :] = 0
    imgGradient[gradNorms > myEpsilon] = (
        imgGradient[gradNorms > myEpsilon] / gradNorms[gradNorms > myEpsilon, np.newaxis]
    )
    x = np.linspace(-1, 1, 2 * stencilSize + 1)
    y = np.linspace(1, -1, 2 * stencilSize + 1)
    result = np.zeros((h, w, 2 * stencilSize + 1, 2 * stencilSize + 1))
    for i in range(h):
        for j in range(w):
            pixelTile = getTile(imgPadded, i, j, stencilSize)
            result[i, j] = pixelTile
            # for factor in np.linspace(-1, 1, 2):
            #     grad = factor * imgGradient[i, j]
            #     iX = np.argmin(np.abs(x - grad[0]))
            #     iY = np.argmin(np.abs(y + grad[1]))
            #     result[i, j, iX, iY] = 0
            grad = imgGradient[i, j]
            iX = np.argmin(np.abs(x - grad[0]))
            iY = np.argmin(np.abs(y + grad[1]))
            result[i, j, iX, iY] = 1
            result[i, j, 2 * stencilSize - iX, 2 * stencilSize - iY] = 1
    return result


img = getTestImageGrayscale(32, 32)
# img = np.array(convertToGrayScale(imgPath))
stencilSize = 1
(nablaPlus, nablaMinus, DxImg, DyImg, DxxImg, DyyImg, DxyImg) = computeSignedDerivatives(img)
filters = twoPointFiltersTest(img, DxImg, DyImg, stencilSize)
h, w, _, _ = filters.shape
print(filters[1, 1].shape)
# fig, ax = plt.subplots()
# im0 = ax.imshow(img)
# im0.set_clim(0, 1)
# ax.axis("off")
nplts = 32
fig2, ax2 = plt.subplots(nplts, nplts, figsize=(9, 9))
plt.subplots_adjust(wspace=0.04, hspace=0.04)
for i in range(nplts):
    for j in range(nplts):
        im = ax2[i, j].imshow(filters[i, j])
        im.set_clim(0, 1)
        ax2[i, j].axis("off")
plt.show()


# img = np.array(convertToGrayScale("IMG_1638_coarse.JPG"))
# stencilSize = 5
# imgPadded = np.pad(img, pad_width=stencilSize, mode="reflect")

# # plt.imshow(imgPadded)
# nImgs = 20
# fig, ax = plt.subplots(nImgs, nImgs, figsize=(9, 9))
# for i in range(nImgs):
#     for j in range(nImgs):
#         tile = getTile(imgPadded, i, j, stencilSize)
#         im = ax[i, j].imshow(tile)
#         im.set_clim(0, 255)
#         ax[i, j].axis("off")
# plt.show()

# endregion

# ---------------------------------------------------------
# region <TEST np.gradient>
# ---------------------------------------------------------

# img = getTestImage(32, 32)
# DyNumpy, DxNumpy = np.gradient(img)
# DyNumpy = -DyNumpy
# DxyNumpy, DxxNumpy = np.gradient(DxNumpy)
# DxyNumpy = -DxyNumpy
# DyyNumpy, DyxNumpy = np.gradient(DyNumpy)
# DyyNumpy = -DyyNumpy

# DxImg, DyImg, DxxImg, DyyImg, DxyImg = computeDerivatives(img)
# fig, ax = plt.subplots(2, 3, figsize=(13, 5))
# im0 = ax[0, 0].imshow(DxImg)
# im1 = ax[0, 1].imshow(DyImg)
# im2 = ax[0, 2].imshow(DxyImg)
# im3 = ax[1, 0].imshow(DxxImg)
# im4 = ax[1, 1].imshow(DyyImg)
# im0.set_clim(-1, 1)
# im1.set_clim(-1, 1)
# im2.set_clim(-1, 1)
# im3.set_clim(-1, 1)
# im4.set_clim(-1, 1)
# plt.show()

# endregion

# ---------------------------------------------------------
# region <TEST GAUSSIAN FILTER>
# ---------------------------------------------------------
# from scipy.ndimage import gaussian_filter
# from parameterValues import *

# img = np.array(convertToGrayScale("IMG_1638_large.JPG"))
# noise = noiseLvl * np.random.randn(*img.shape)
# img += noise

# sigma = 0.5

# imgDenoised = gaussian_filter(img, sigma)
# fig, ax = plt.subplots()
# plot = ax.imshow(imgDenoised, cmap="viridis")
# ax.axis("off")
# plot.set_clim(0, 1)
# plot.figure.savefig(f"results/gauss{sigma}.png", dpi=300, bbox_inches="tight", pad_inches=0)

# plt.show()

# endregion
