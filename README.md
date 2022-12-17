# Face Image Similarity OpenCV
A simple project on measuring two face images similarities using Cosine and Euclidean distance

This project involves building a facial recognition model that compares two images and returns a similarity score. The model follows the following steps:

## Preprocessing

The input images are preprocessed to ensure that they are in a consistent format. This typically involves:

- Resizing the images to the same size
- Converting the images to grayscale
- Applying other types of image enhancement techniques, as needed
## Feature Extraction

Features are extracted from the preprocessed images using:

- Haar Cascade Classifier
## Similarity Measurement

The similarity between the extracted features is calculated using a distance measure such as:

- Euclidean distance
- Cosine similarity

## Output

The calculated similarity score is returned. The score can be returned as is, or mapped to a certain range (such as 0 to 1) to indicate the level of similarity between the two images.

## Considerations

Building a facial recognition model involves several technical challenges and there are many factors that can affect its performance. It is important to carefully consider the choice of:

- Preprocessing techniques
- Feature extraction methods
- Similarity measure
- Training data quality and diversity
It is also important to test the model on a large and diverse set of images to ensure that it performs well in a variety of scenarios.
