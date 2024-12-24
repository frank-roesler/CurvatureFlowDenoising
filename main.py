from numericsTools import *
from imageProcessing import *
from parameterValues import *
from functools import partial

if __name__ == "__main__":
    img = np.array(convertToGrayScale(imgPath))
    noise = noiseLvl * np.random.randn(*img.shape)

    # flowFunction = partial(gradientModulatedMeanCurvFlow, VGradient=VGradient)
    flowFunction = partial(meanCurvModulatedFlow, th1=M1, th2=M2)
    # flowFunction = partial(doubleModulatedFlow, VGradient=VGradient, th1=M1, th2=M2)
    # flowFunction = partial(doubleModulatedFlowReversed, VGradient=VGradient, th1=M1, th2=M2)

    denoisedImage = solveLevelSetEquationGrayscale(
        img + noise,
        stencil,
        iters,
        dt,
        flowFunction,
        plot=True,
        plottingFreq=plottingFreq,
        orig=img,
        savePlots=False,
    )
    plt.show()
