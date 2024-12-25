# CurvatureFlowDenoising
Python Implementation of a classical image denoising method based on curvature flow. 
Introduced in [1], [2].

To use this code base, install `requirements.txt` via PIP and specify the image path in `parameterValues.py`. 
Currently, the main file adds Gaussian noise to the image and then runs the min/max algorithm to remove it. If the image is alredy noisy, this part should be removed from `main.py`.

[1] *Malladi, R. and Sethian, J.A., 1995. Image processing via level set curvature flow. proceedings of the National Academy of sciences, 92(15), pp.7046-7050.*  
[2] *Malladi, R. and Sethian, J.A., 1996. Image processing: Flows under min/max curvature and mean curvature. Graphical models and image processing, 58(2), pp.127-141.*
