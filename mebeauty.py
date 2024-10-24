import os

from glob import glob

import cv2

import numpy as np

from visual_attractiveness import VisualAttractiveness

predictor = VisualAttractiveness()


def load_images_from_folder(folder_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/MEBeauty-database-main/original_images'):
    images = {}
    for filename in [y for x in os.walk(folder_name) for y in glob(os.path.join(x[0], '*.jpg'))]:
        img = cv2.imread(filename)

        if img is not None:
            images[filename] = img

    return images


def load_scores_from_file(file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/MEBeauty-database-main/scores/generic_scores_all.csv'):
    scores = {}
    for line in open(file_name, 'r').readlines():
        split = line.split(';')
        if len(split) >= 3:
            try:
                scores[split[2]] = float(split[0].replace(',', '.'))
            except ValueError:
                continue

    return scores


def test(image_with_score):
    avg_face = []
    avg_body = []
    for name, image, score in image_with_score:
        facial_beauty_score, _, _ = predictor.predict_facial_beauty(image, [["attractive", "unattractive"]])
        human_beauty_score, _, _ = predictor.predict_physical_beauty(image, [["attractive", "unattractive"]])

        if facial_beauty_score is not None and human_beauty_score is not None:
            avg_face.append(abs(facial_beauty_score - (score - 1) * 10/9))
            avg_body.append(abs(human_beauty_score - (score - 1) * 10/9))

    print(np.mean(np.array(avg_face)))
    print(np.mean(np.array(avg_body)))


if __name__ == "__main__":
    name_to_score = load_scores_from_file()
    name_to_image = load_images_from_folder()

    image_with_score = [(name, name_to_image[name], score) for name, score in name_to_score.items() if name in list(name_to_image.keys())]

    test(image_with_score)



