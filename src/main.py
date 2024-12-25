from lib.numericsTools import *
from lib.imageProcessing import *
from parameterValues import *
from functools import partial

if __name__ == "__main__":
    img = np.array(convertToGrayScale(imgPath))
    noise = noiseLvl * np.random.randn(*img.shape)
    noisyImg = img + noise

    # flowFunction = partial(gradientModulatedMeanCurvFlow, VGradient=VGradient)
    # flowFunction = partial(doubleModulatedFlow, VGradient=VGradient, th1=M1, th2=M2)
    # flowFunction = partial(doubleModulatedFlowReversed, VGradient=VGradient, th1=M1, th2=M2)
    flowFunction = partial(meanCurvModulatedFlow, th1=M1, th2=M2)

    denoisedImage = solveLevelSetEquationGrayscale(
        noisyImg,
        stencil,
        iters,
        dt,
        flowFunction,
        plot=plotDuringLoop,
        plottingFreq=plottingFreq,
        groundTruth=img,
        savePlots=savePlots,
    )
    # plt.show()
