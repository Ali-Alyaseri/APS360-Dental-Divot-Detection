
dental-caries - v1 2023-08-15 4:39pm
==============================

This dataset was exported via roboflow.com on August 15, 2023 at 12:41 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 5527 images.
Caries are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random rotation of between -23 and +23 degrees
* Random shear of between -15° to +15° horizontally and -24° to +24° vertically
* Random brigthness adjustment of between -32 and +32 percent
* Random exposure adjustment of between -15 and +15 percent
* Salt and pepper noise was applied to 5 percent of pixels


