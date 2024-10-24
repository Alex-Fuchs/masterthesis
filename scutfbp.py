import os

import cv2

import numpy as np

from visual_attractiveness import VisualAttractiveness

predictor = VisualAttractiveness()


def load_images_from_folder(folder_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/Images'):
    images = {}
    for filename in os.listdir(folder_name):
        img = cv2.imread(os.path.join(folder_name, filename))
        if img is not None:
            images[filename] = img
    return images


def load_scores_from_file(file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/train_test_files/All_labels.txt'):
    scores = {}
    for line in open(file_name, 'r').readlines():
        split = line.split()
        scores[split[0]] = float(split[1])
    return scores


def test(image_with_score):
    avg_face = []
    avg_body = []
    for name, image, score in image_with_score:
        facial_beauty_score, _, _ = predictor.predict_facial_beauty(image, [["attractive", "unattractive"]])
        human_beauty_score, _, _ = predictor.predict_physical_beauty(image, [["attractive", "unattractive"]])

        if facial_beauty_score is not None and human_beauty_score is not None:
            avg_face.append(abs(facial_beauty_score - (score - 1) * 2.5))
            avg_body.append(abs(human_beauty_score - (score - 1) * 2.5))

    print(np.mean(np.array(avg_face)))
    print(np.mean(np.array(avg_body)))


if __name__ == "__main__":
    name_to_score = load_scores_from_file()
    name_to_image = load_images_from_folder()

    image_with_score = [(name, image, name_to_score[name]) for name, image in name_to_image.items()]

    names = ['CF' + str(i) + '.jpg' for i in range(650, 750)] + ['CM' + str(i) + '.jpg' for i in range(650, 750)]
    result = [(name, image, score) for name, image, score in image_with_score if name in names]

    test(result)



