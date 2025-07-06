# SMILE: Smart Machine-learning Imaging for Lesion Evaluation

## Overview

**SMILE** is a machine learning-based dental assistant designed to automatically detect cavities (caries) in panoramic dental X-ray images. It aims to improve early cavity diagnosis, support dentists with reliable second opinions, and streamline routine dental screenings.

## Motivation

Cavities affect over 2 billion people worldwide. Diagnosis using panoramic dental X-rays can be time-consuming and prone to error due to variability in image quality and practitioner experience. SMILE leverages computer vision to detect cavities efficiently and accurately, improving diagnostic consistency and reducing the burden on healthcare professionals.

## Key Features

- **Panoramic X-ray Input**  
  Works with real-world dental radiographs from multiple public datasets.

- **Object Detection with YOLOv8**  
  Locates and identifies cavities with bounding boxes and confidence scores.

- **Baseline Comparison**  
  Implements a traditional HOG + SVM classifier for benchmarking performance.

- **End-to-End Processing**  
  Complete image preprocessing, annotation conversion, training, and evaluation pipeline.

- **Interpretable and Ethical AI**  
  Predictions are visualized clearly, with strong emphasis on data privacy and clinical support.

## Datasets

Data is sourced from three publicly available datasets:
- **Kaggle**: Panoramic X-rays with bounding box annotations.
- **Mendeley Data**: Radiographs in YOLO format.
- **Synapse DC1000**: High-resolution images with segmentation masks converted to bounding boxes.

All datasets are standardized to YOLO format, with unified image dimensions and consistent class labeling.

## Architecture

The core of SMILE is a YOLOv8-based object detection model consisting of:

- **Backbone**: CNN pre-trained on large-scale datasets for feature extraction.
- **Neck**: PANet-style architecture for multi-scale feature fusion.
- **Head**: Predicts bounding boxes, objectness scores, and cavity class probabilities.

## Baseline Model

A classical machine learning approach is used as a benchmark:

- **Feature Extraction**: Histogram of Oriented Gradients (HOG)
- **Classification**: Support Vector Machine (SVM)
- **Post-processing**: Non-Maximum Suppression to refine overlapping detections

This pipeline offers an interpretable and lightweight baseline for comparison.

## Ethical Considerations

- SMILE is designed as a **decision support tool**, not a diagnostic replacement.
- **False positives** may lead to unnecessary procedures, and **false negatives** may delay treatment.
- Personal health data is anonymized, with attention to privacy, consent, and transparency.
- Model outputs include interpretable visual markers (e.g., bounding boxes) to aid clinical review.

## Citation

If you use or build upon this work, please cite:

> SMILE: Smart Machine-learning Imaging for Lesion Evaluation. 2025.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Contributors**:
- Student 1  
- Student 2  
- Student 3  
- Student 4

