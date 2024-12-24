myEpsilon = 1e-8

imgPath = "IMG_1638_large.JPG"
dt = 0.1
iters = 500
stencil = 1
noiseLvl = 0.2
plottingFreq = 10

# Threshold for gradient modulation
VGradient = 0.02

# Thresholds for mean curvature modulation
# (nothing < M1 < min/max flow < M2 < mean curvature flow)
M1 = 0.01
M2 = 0.7
