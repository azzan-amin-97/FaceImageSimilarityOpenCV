import cv2

from utils import compare_img_and_calc_similarity
path_img_1 = 'data/img/face1.jpg'
path_img_2 = 'data/img/non-face2.jpg'

image1 = cv2.imread(path_img_1)
image2 = cv2.imread(path_img_2)
images = [image1, image2]

if __name__ == "__main__":
    print('\n#####################################')
    print('Similarity Score between two images')
    print('####################################\n')

    print('Cosine similarity score:')
    similarity_score = compare_img_and_calc_similarity(images, score_type='cosine')
    print("Similarity score (cosine): {:.2f}%".format(similarity_score))
    print('\n')

    print('Euclidean distance similarity score:')
    similarity_score = compare_img_and_calc_similarity(images, score_type='euclidean')
    print("Similarity score (euclidean): {:.2f}%".format(similarity_score))

    print('\n####################################\n')

