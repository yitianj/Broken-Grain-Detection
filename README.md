# Broken-Grain-Detection

## Discription

Grain breakage is an important factor that affects grain yield and quality. It is of great significance to do research on rice grain breakage identification and detection methods to promote green, high-yield, stable, high-quality and efficient development of rice. This project aims to automate the process of broken-graph-detection. Specifically, the digital image processing technology is used to extract the morphological feature parameters. Among various machine learning algorithms, the decision tree and random forests are proven to be simple but effective models.

## Method

### Image processing module

* Image pre-processing (singleimg.py). We first pre-process the raw image using the OpenCV library including grayscale, image enhancement, image segmentation, image denoising, and edge detection. 

<figure><img src="./result/decision tree.png" height=100px></img><figcaption>Original Grain</figcaption></figure>
<figure><img src="./result/decision tree.png" height=100px></img><figcaption>Pre-processed Grain</figcaption></figure>


* Feature Extraction that extracts morphological features such as rice grain sample area, perimeter, roundness, long axis, short axis, aspect ratio, rectangularity, density, etc. 

### Broken-graph detection (classification) module
Considering the training data
2. Using the decision tree, the Gini coefficient is selected as the segmentation metric of the decision tree node split, supplemented by the method of random forest for comparison, and finally the identification model of broken rice kernels is established.

## Data

We collect xxx. Talk about annotation(#labled). Specifications: random forest (#trees, #depth). Decision Tree

## Result and Analysis:
<img src="./result/decision tree.png" height=300px></img>
