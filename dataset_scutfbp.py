import os

import cv2

import torch
import numpy as np

from visual_attractiveness import VisualAttractiveness


def load_image_path_to_score(folder_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/Images', file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/train_test_files/All_labels.txt'):
    image_path_to_score = {}
    for line in open(file_name, 'r').readlines():
        splitted_line = line.split()

        image_path_to_score[os.path.join(folder_name, splitted_line[0])] = (float(splitted_line[1]) - 1) * 2.5

    names = ([folder_name + '/CF' + str(i) + '.jpg' for i in range(650, 750)] + [folder_name + '/CM' + str(i) + '.jpg' for i in range(650, 750)])
    image_path_to_score = [(image_path, score) for image_path, score in image_path_to_score.items() if image_path in names]

    return image_path_to_score


def test(predictor, image_path_to_score):
    avg_body = []
    for image_path, score in image_path_to_score:
        bgr = cv2.imread(image_path)

        human_beauty_score, _ = predictor.predict_physical_beauty(bgr)

        if human_beauty_score is not None:
            avg_body.append(abs(human_beauty_score - score))

    print(np.mean(np.array(avg_body)))


if __name__ == "__main__":
    text_features = torch.load("weights/text_features_mebeauty_100k.pth")
    text_features = torch.tensor(text_features, dtype=torch.float32)

    predictor = VisualAttractiveness()

    image_path_to_score = load_image_path_to_score()

    test(predictor, image_path_to_score)



