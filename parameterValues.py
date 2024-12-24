myEpsilon = 1e-8

imgPath = "IMG_1638.JPG"
dt = 0.1  # time stepp for PDE solver
iters = 100
stencil = 1  # larger value leads to more smoothing (theoretically...)
noiseLvl = 0.1
plottingFreq = 50
plotDuringLoop = True
savePlots = True
imagePadding = "constant"  # must work for np.pad and scipy.ndimage.convolve

# Threshold for gradient modulation
VGradient = 0.02

# Thresholds for mean curvature modulation
# (nothing < M1 < min/max flow < M2 < mean curvature flow)
M1 = 0.01
M2 = 0.7
