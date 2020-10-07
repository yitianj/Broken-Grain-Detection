# Broken-Grain-Detection

## Discription

Grain breakage is an important factor that affects rice yield and quality. It is of great significance to do research on rice grain breakage identification and detection methods to promote green, high-yield, stable, high-quality and efficient development of rice. The digital image processing technology is used to extract the morphological feature parameters. The decision tree and random forest are used to build models respectively. 

## Inplementation

1. I used OpenCV to do the image pre-processing including grayscale, image enhancement, image segmentation, image denoising, and edge detection. And extracted 8 morphological features such as rice grain sample area, perimeter, roundness, long axis, short axis, aspect ratio, rectangularity, density, etc. 
2. Using the decision tree, the Gini coefficient is selected as the segmentation metric of the decision tree node split, supplemented by the method of random forest for comparison, and finally the identification model of broken rice kernels is established.

## Result:
