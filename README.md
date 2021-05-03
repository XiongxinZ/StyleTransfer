# StyleTransfer

## Description

Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but "painted" in the style of the style reference image.

This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.

## Data

[Celebrity Photos](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

[Visual art encyclopedia Wikiart](https://www.wikiart.org/)

## Methods

### Method 1

#### Implementation
Run Model1/style_transfer.ipynb.
The first cell gives the notebook access to google drive. So if the datasets are put in the google drive, run the first cell first and get access. After that, run all cells left and get results.
You may change the file path to get more different results with different styles and contents.

#### Structure of Model
1 Has a function that cut images from the center, a function that load and resize images and a function that display images.
The model behind this images is arbitrary neural artistic stylization network. It consists of style transfer network and style prediction network. There are two kinds of losses: style loss and content loss. The model use convolutional networks to minimize the total loss and produces stylized images.

#### Results
Shown in the picture are contents of faces with some noise in the background, profile, bodies and face with a clear background. For styles, we choose abstract, impressionism, cubism baroque and rococo.
![](./result/m1.png)

#### Discussion
The CNN model works best when there is a front face with a clear background. For different styles, baroque generates the best results because the average loss is the smallest. When there is a profile or a body on the image, the result is not so clear compared to faces. Also, the model does not work on cubism as well as on other styles.

### Method 2

#### Implementation

#### Structure of Model

#### Results

#### Discussion

### Method 3

#### Implementation

#### Structure of Model

#### Results

#### Discussion

## Paper

[Art Nouveau Style Transfer with Face Alignment](http://cs230.stanford.edu/projects_fall_2019/reports/26261057.pdf)

[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/pdf/1705.06830.pdf)

## Tutorial

[Fast Style Transfer for Arbitrary Styles](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)

[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://github.com/magenta/magenta/tree/master/magenta/models/arbitrary_image_stylization)

[Tensorflow_image_stylization](https://github.com/Robinatp/Tensorflow_image_stylization)

https://paperswithcode.com/paper/exploring-the-structure-of-a-real-time
