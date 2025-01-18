import os

import cv2

import numpy as np

from visual_attractiveness import VisualAttractiveness


def load_image_path_to_image(folder_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/Images'):
    image_path_to_image = {}
    for filename in os.listdir(folder_name):
        img = cv2.imread(os.path.join(folder_name, filename))

        if img is not None:
            image_path_to_image[filename] = img

    return image_path_to_image


def load_image_path_to_score(file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/train_test_files/All_labels.txt'):
    image_path_to_score = {}
    for line in open(file_name, 'r').readlines():
        splitted_line = line.split()

        image_path_to_score[splitted_line[0]] = (float(splitted_line[1]) - 1) * 2.5

    return image_path_to_score


def test(predictor, image_to_score):
    avg_body = []
    for name, image, score in image_to_score:
        human_beauty_score, _ = predictor.predict_physical_beauty(image)

        if human_beauty_score is not None:
            avg_body.append(abs(human_beauty_score - score))

    print(np.mean(np.array(avg_body)))


if __name__ == "__main__":
    predictor = VisualAttractiveness()

    image_path_to_score = load_image_path_to_score()
    image_path_to_image = load_image_path_to_image()

    image_to_score = [(name, image, image_path_to_score[name]) for name, image in image_path_to_image.items()]

    names = ['CF' + str(i) + '.jpg' for i in range(650, 750)] + ['CM' + str(i) + '.jpg' for i in range(650, 750)]
    image_to_score = [(image, score) for name, image, score in image_to_score if name in names]

    test(predictor, image_to_score)



