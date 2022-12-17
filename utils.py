import cv2
from scipy.spatial.distance import euclidean
import numpy as np
# import torch
# import torchvision.models as models

def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (200, 200))
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0] * (target_len - len(some_list))


def get_target_vector_length(f1, f2):
    if len(f1) > len(f2):
        return len(f1)
    else:
        return len(f2)


def min_max_distance(f1, f2, n, distance_type='min'):
    l = f1 + f2
    if distance_type == 'min':
        v = [0] * n
        return v
    elif distance_type == 'max':
        v = [max(l)] * n
        return v



# def extract_features_cnn(image):
#     # Convert the image to a tensor
#     image = torch.from_numpy(image).float()
#     image = image.unsqueeze(0)

#     # Load a pre-trained CNN model
#     model = models.resnet50(pretrained=True)

#     # Extract the features from the model's second last layer
#     features = model(image)
#     features = features.squeeze(0)

#     return features

# def extract_features(image, feature_extraction='haarcascade'):
#     features = []
#     # if feature_extraction=='cnn':
#     #     features = extract_features_cnn(image)
#     # if feature_extraction=='lbp':
#     #     features = extract_features_lbp(image)
#     if feature_extraction=='haarcascade':
#         features = extract_features_haarcascade(image)
#     return features

def extract_features_haarcascade(image):
    # Load the cascade classifiers
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_mcs_mouth.xml')
    left_eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_righteye_2splits.xml")

    # Detect the facial features in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return False

    for (x, y, w, h) in faces:
        roi_gray = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)

    # Extract the feature vectors from the facial features
    features = []
    result = []
    for (ex, ey, ew, eh) in eyes:
        features.extend([ex, ey, ew, eh])
    for (lex, ley, lew, leh) in left_eye:
        features.extend([lex, ley, lew, leh])
    for (rex, rey, rew, reh) in right_eye:
        features.extend([rex, rey, rew, reh])
    for (nx, ny, nw, nh) in nose:
        features.extend([nx, ny, nw, nh])
    for (mx, my, mw, mh) in mouth:
        features.extend([mx, my, mw, mh])

    return features

def extract_features_lbp(image):
    # # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the LBPs for the image
    lbp = cv2.localBinaryPattern(image, 8, 1, method="uniform")

    # Flatten the LBPs into a 1D array
    features = lbp.flatten()
    
    return features

def calculate_cosine_similarity(features1, features2):
    target_len = get_target_vector_length(features1, features2)
    features1 = pad_or_truncate(features1, target_len)
    features2 = pad_or_truncate(features2, target_len)
    # Calculate the dot product between the feature vectors
    dot_product = np.dot(features1, features2)

    # Calculate the magnitudes of the feature vectors
    magnitudes = np.linalg.norm(features1) * np.linalg.norm(features2)

    # Calculate the cosine similarity score
    similarity_score = dot_product / magnitudes

    return similarity_score

def calculate_euclidean_distance(features1, features2):
    # Get the max vector length between features 1 and 2
    target_len = get_target_vector_length(features1, features2)
    # Standardize vector by adding pads to shorter vector
    features1 = pad_or_truncate(features1, target_len)
    features2 = pad_or_truncate(features2, target_len)
    # Calculate the Euclidean distance between the feature vectors
    distance = np.linalg.norm(np.array(features1) - np.array(features2))
    # Calculate the maximum possible distance between the feature vectors
    minv = min_max_distance(features1, features2, target_len, distance_type='min')
    maxv = min_max_distance(features1, features2, target_len, distance_type='max')
    # max_distance = np.linalg.norm(np.max(features1) - np.min(features1)) + np.linalg.norm(np.max(features2) - np.min(features2))
    max_distance = euclidean(minv,maxv)
    # Calculate the similarity score in percentage
    similarity_score = (max_distance - distance) / max_distance

    return similarity_score


def calculate_similarity(features1, features2, score_type='cosine'):
    target_len = get_target_vector_length(features1, features2)
    features1 = pad_or_truncate(features1, target_len)
    features2 = pad_or_truncate(features2, target_len)
    
    similarity_score = 0.0

    if score_type == 'euclidean':
        # Calculate the Euclidean distance between the feature vectors
        distance = euclidean(features1, features2)
        print(features1,'\n',features2)

        # Normalize the distance by dividing it by the maximum possible distance
        minv = min_max_distance(features1, features2, target_len, distance_type='min')
        maxv = min_max_distance(features1, features2, target_len, distance_type='max')
        max_distance = euclidean(minv,maxv)

        print('Distance between two images:',distance)
        print('Image vector max distance:', max_distance)

        similarity_score = ((max_distance - distance) / max_distance) * 100
        print('Score type =>', score_type)

    elif score_type == 'cosine':
        target_len = get_target_vector_length(features1, features2)
        features1 = pad_or_truncate(features1, target_len)
        features2 = pad_or_truncate(features2, target_len)
        # Calculate the dot product between the feature vectors
        dot_product = np.dot(features1, features2)

        # Calculate the magnitudes of the feature vectors
        magnitudes = np.linalg.norm(features1) * np.linalg.norm(features2)

        # Calculate the cosine similarity score
        similarity_score = dot_product / magnitudes
    return similarity_score


def compare_img_and_calc_similarity(images, feature_extraction='haarcascade', score_type='cosine'):
    # Preprocess the input images
    preprocessed_img1 = preprocess_image(images[0])
    preprocessed_img2 = preprocess_image(images[1])

    # Extract features from the preprocessed images using Haar cascades
    features1 = extract_features_haarcascade(preprocessed_img1)
    features2 = extract_features_haarcascade(preprocessed_img2)

    if not features1:
        print("No face detected in Image 1")
    else:
        print("Face detected in Image 1")
    if not features2:
        print("No face detected in Image 2")
    else:
        print("Face detected in Image 2")

    if (features1==False) or (features2==False):
        print('Non-face image detected, unable to proceed for similarity measures. Exited')
        return 0


    # Calculate the similarity between the extracted features
    if score_type=='euclidean':
        similarity_score = calculate_euclidean_distance(features1, features2)
    elif score_type=='cosine':
        similarity_score = calculate_cosine_similarity(features1, features2)
    else:
        raise ValueError('Invalid score type')

    # Map the similarity score to a certain range (such as 0 to 1)
    mapped_score = map_score(similarity_score, score_type=score_type)

    return mapped_score

def map_score(score, score_type='euclidean'):
    # Map the score to a certain range (such as 0 to 1)
    print('score:', score)
    mapped_score = 0.0
    if score_type in ['cosine','euclidean']:
        mapped_score = score * 100
    else:
        raise ValueError('Invalid score type')

    print('mapped_score:', mapped_score)
    return mapped_score
