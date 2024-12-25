# CurvatureFlowDenoising
Python Implementation of a classical image denoising method based on curvature flow. 
Introduced in [R. Malladi, J.A. Sethian, Image processing via level set curvature flow., Proc. Natl. Acad. Sci. U.S.A. 92 (15) 7046-7050, https://doi.org/10.1073/pnas.92.15.7046 (1995).]

To use this code base, install `requirements.txt` via PIP and specify the image path in `parameterValues.py`. 
Currently, the main file adds Gaussian noise to the image and then runs the min/max algorithm to remove it. If the image is alredy noisy, this part should be removed from `main.py`.
