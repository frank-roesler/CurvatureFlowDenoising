import numpy as np
from lib.imageProcessing import *
from lib.utils import *
from parameterValues import myEpsilon


def computeDerivatives(img: np.array) -> tuple[np.array, np.array, np.array, np.array, np.array]:
    img = np.pad(img, pad_width=1, mode=imagePadding)

    imgPlusX = np.roll(img, 1, axis=1)
    imgMinusX = np.roll(img, -1, axis=1)
    imgPlusY = np.roll(img, -1, axis=0)
    imgMinusY = np.roll(img, 1, axis=0)

    imgPlusPlus = np.roll(imgPlusX, -1, axis=0)
    imgMinusMinus = np.roll(imgMinusX, 1, axis=0)
    imgPlusMinus = np.roll(imgPlusX, 1, axis=0)
    imgMinusPlus = np.roll(imgMinusX, -1, axis=0)

    DxImgPadded = (imgMinusX - imgPlusX) / 2
    DyImgPadded = (imgMinusY - imgPlusY) / 2
    DxxImgPadded = imgMinusX - 2 * img + imgPlusX
    DyyImgPadded = imgMinusY - 2 * img + imgPlusY
    DxyImgPadded = (imgPlusPlus - imgPlusMinus - imgMinusPlus + imgMinusMinus) / 4

    DxImg = DxImgPadded[1:-1, 1:-1]
    DyImg = DyImgPadded[1:-1, 1:-1]
    DxxImg = DxxImgPadded[1:-1, 1:-1]
    DyyImg = DyyImgPadded[1:-1, 1:-1]
    DxyImg = DxyImgPadded[1:-1, 1:-1]
    return DxImg, DyImg, DxxImg, DyyImg, DxyImg


def computeSignedDerivatives(
    img: np.array,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    img = np.pad(img, pad_width=1, mode=imagePadding)

    imgPlusX = np.roll(img, 1, axis=1)
    imgMinusX = np.roll(img, -1, axis=1)
    imgPlusY = np.roll(img, -1, axis=0)
    imgMinusY = np.roll(img, 1, axis=0)

    imgPlusPlus = np.roll(imgPlusX, -1, axis=0)
    imgMinusMinus = np.roll(imgMinusX, 1, axis=0)
    imgPlusMinus = np.roll(imgPlusX, 1, axis=0)
    imgMinusPlus = np.roll(imgMinusX, -1, axis=0)

    DxPlus = imgMinusX - img
    DxMinus = img - imgPlusX
    DyPlus = imgMinusY - img
    DyMinus = img - imgPlusY

    nablaPlus = np.sqrt(
        np.maximum(DxMinus, 0) ** 2
        + np.minimum(DxPlus, 0) ** 2
        + np.maximum(DyMinus, 0) ** 2
        + np.minimum(DyPlus, 0) ** 2
    )
    nablaMinus = np.sqrt(
        np.maximum(DxPlus, 0) ** 2
        + np.minimum(DxMinus, 0) ** 2
        + np.maximum(DyPlus, 0) ** 2
        + np.minimum(DyMinus, 0) ** 2
    )

    DxImgPadded = (imgMinusX - imgPlusX) / 2
    DyImgPadded = (imgMinusY - imgPlusY) / 2
    DxxImgPadded = imgMinusX - 2 * img + imgPlusX
    DyyImgPadded = imgMinusY - 2 * img + imgPlusY
    DxyImgPadded = (imgPlusPlus - imgPlusMinus - imgMinusPlus + imgMinusMinus) / 4

    DxImg = DxImgPadded[1:-1, 1:-1]
    DyImg = DyImgPadded[1:-1, 1:-1]
    DxxImg = DxxImgPadded[1:-1, 1:-1]
    DyyImg = DyyImgPadded[1:-1, 1:-1]
    DxyImg = DxyImgPadded[1:-1, 1:-1]
    nablaPlus = nablaPlus[1:-1, 1:-1]
    nablaMinus = nablaMinus[1:-1, 1:-1]

    return (nablaPlus, nablaMinus, DxImg, DyImg, DxxImg, DyyImg, DxyImg)


def curvature(img: np.array) -> tuple[np.array, np.array, np.array]:
    # 3/2 cancels to 1 by leaving out |\nabla phi| in level set eq.
    (nablaPlus, nablaMinus, DxImg, DyImg, DxxImg, DyyImg, DxyImg) = computeSignedDerivatives(img)
    meanCurv = ((1 + DxImg**2) * DyyImg + (1 + DyImg**2) * DxxImg - 2 * DxImg * DyImg * DxyImg) / (
        DxImg**2 + DyImg**2 + 1
    ) ** (3 / 2)
    curv = (DxImg**2 * DyyImg + DyImg**2 * DxxImg - 2 * DxImg * DyImg * DxyImg) / (
        DxImg**2 + DyImg**2 + 1e-16
    ) ** (3 / 2)
    curv[DxImg**2 + DyImg**2 < myEpsilon] = 0
    return curv, DxImg, DyImg, meanCurv


def F(kappa: np.array, treshold: np.array, img: np.array, stencilSize: int) -> np.array:
    result = np.zeros(img.shape)
    avgs = localAverage(img, stencilSize)
    area = avgs < treshold
    result[area] = np.maximum(kappa[area], 0)  # * nablaPlus[area]
    result[~area] = np.minimum(kappa[~area], 0)  # * nablaMinus[~area]
    return result, area


# def solveLevelSetEquationBinary(
#     img: np.array,
#     stencilSize: int,
#     iterations: int,
#     dt: float,
#     plot: bool = False,
#     plottingFreq: int = 100,
#     orig: np.array = None,
# ) -> np.array:
#     """NEEDS TO BE UPDATED!!! DO NOT USE!!!."""
#     result = img
#     diffs = []
#     errs = []
#     ax, im2 = initPlots(result, plot)
#     for it in range(iterations):
#         FImg = F(0.5, result, stencilSize)
#         result = result + dt * FImg  # * np.sqrt(DxImg2 + DyImg2)
#         diffs.append(np.sqrt(np.mean(FImg**2)))
#         errs.append(np.sqrt(np.mean((orig - np.clip(result, 0, 1)) ** 2)))
#         updatePlots(ax, im2, result, errs, diffs, plot, it, plottingFreq)
#     return result


def gradientModulatedMeanCurvFlow(DxImg, DyImg, FImg, meanCurv, VGradient):
    rhs = np.zeros(DxImg.shape)
    absGrad = np.sqrt(DxImg**2 + DyImg**2)
    gradientSmallArea = absGrad < VGradient
    rhs[~gradientSmallArea] = FImg[~gradientSmallArea] * absGrad[~gradientSmallArea]
    rhs[gradientSmallArea] = meanCurv[gradientSmallArea] * np.sqrt(
        1 + DxImg[gradientSmallArea] ** 2 + DyImg[gradientSmallArea] ** 2
    )
    return rhs, gradientSmallArea


def meanCurvModulatedFlow(DxImg, DyImg, FImg, meanCurv, th1, th2):
    rhs = np.zeros(DxImg.shape)
    absGrad = np.sqrt(DxImg**2 + DyImg**2)
    absMeanCurc = np.abs(meanCurv)
    midRange = (absMeanCurc > th1) & (absMeanCurc <= th2)
    highRange = absMeanCurc > th2
    rhs[midRange] = FImg[midRange] * absGrad[midRange]
    rhs[highRange] = meanCurv[highRange] * np.sqrt(1 + DxImg[highRange] ** 2 + DyImg[highRange] ** 2)
    return rhs, 0.5 * midRange + highRange


def doubleModulatedFlow(DxImg, DyImg, FImg, meanCurv, th1, th2, VGradient):
    rhs = np.zeros(DxImg.shape)
    absGrad = np.sqrt(DxImg**2 + DyImg**2)
    gradientSmallArea = absGrad < VGradient
    flow, ranges = meanCurvModulatedFlow(DxImg, DyImg, FImg, meanCurv, th1, th2)
    rhs[~gradientSmallArea] = flow[~gradientSmallArea]
    rhs[gradientSmallArea] = meanCurv[gradientSmallArea] * np.sqrt(
        1 + DxImg[gradientSmallArea] ** 2 + DyImg[gradientSmallArea] ** 2
    )
    return rhs, gradientSmallArea


def doubleModulatedFlowReversed(DxImg, DyImg, FImg, meanCurv, th1, th2, VGradient):
    rhs = np.zeros(DxImg.shape)
    absMeanCurc = np.abs(meanCurv)
    midRange = (absMeanCurc > th1) & (absMeanCurc <= th2)
    highRange = absMeanCurc > th2
    flow, area = gradientModulatedMeanCurvFlow(DxImg, DyImg, FImg, meanCurv, VGradient)
    rhs[midRange] = flow[midRange]
    rhs[highRange] = meanCurv[highRange] * np.sqrt(1 + DxImg[highRange] ** 2 + DyImg[highRange] ** 2)
    return rhs, 0.5 * midRange + highRange


def solveLevelSetEquationGrayscale(
    img: np.array,
    stencilSize: int,
    iterations: int,
    dt: float,
    flowFunction,
    plot: bool = False,
    plottingFreq: int = 100,
    groundTruth: np.array = None,
    savePlots: bool = False,
) -> np.array:
    """Solves equation dphi/dt = F|grad(phi)| for given number of iterations.
    Image must be binary with values in {0,1} only."""
    diffs = []
    errs = []
    result = img
    # result = np.clip(img, 0, 1)
    ax, myPlots = initPlots(result, groundTruth, plot, savePlots)
    for it in range(iterations + 2):
        kappa, DxImg, DyImg, meanCurv = curvature(result)
        thresholds = applyTwoPointFilters(result, DxImg, DyImg, stencilSize)
        FImg, area = F(kappa, thresholds, result, stencilSize)
        flowFct, modulationArea = flowFunction(DxImg, DyImg, FImg, meanCurv)
        result += dt * flowFct

        diffs.append(np.sqrt(np.mean(flowFct**2)))
        errs.append(np.sqrt(np.mean((groundTruth - np.clip(result, 0, 1)) ** 2)))
        updatePlots(ax, myPlots, result, modulationArea, errs, diffs, plot, it, plottingFreq, savePlots)

    return result
