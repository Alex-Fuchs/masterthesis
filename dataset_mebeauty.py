import os
from glob import glob

import cv2

import numpy as np

from visual_attractiveness import VisualAttractiveness


def load_image_path_to_latent(filenames=None, folder_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/MEBeauty-database-main/original_images'):
    predictor = VisualAttractiveness()

    if filenames is None:
        filenames = [y for x in os.walk(folder_name) for y in glob(os.path.join(x[0], '*.jpg'))]

    image_path_to_latent = {}
    for filename in filenames:
        bgr = cv2.imread(filename)

        if bgr is not None:
            human = predictor.get_human(bgr)

            if human is not None:
                latent = predictor.encode_image(human)

                image_path_to_latent[filename] = latent

    return image_path_to_latent


def load_image_path_to_score(file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/MEBeauty-database-main/scores/generic_scores_all.csv'):
    image_path_to_score = {}
    for line in open(file_name, 'r').readlines():
        splitted_line = line.split(';')

        image_path = splitted_line[2]

        if os.path.exists(image_path):
            try:
                score = splitted_line[0]
                score = score.replace(',', '.')
                score = float(score)
                score -= 1
                score *= 10/9
            except ValueError:
                continue

            image_path_to_score[image_path] = score

    return image_path_to_score


def load_image_path_to_personalized_scores(file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/MEBeauty-database-main/scores/generic_scores_all.csv'):
    image_path_to_scores = {}

    column_names = open(file_name, 'r').readlines()[0]
    column_names = column_names.split(';')

    for line in open(file_name, 'r').readlines()[1:]:
        splitted_line = line.split(';')

        image_path = splitted_line[2]

        if os.path.exists(image_path):
            scores_with_raters_of_image = {}
            for column_name, score in zip(column_names[3:], splitted_line[3:]):
                if score != '':
                    try:
                        scores_with_raters_of_image[column_name] = (float(score.replace(',', '.')) - 1) * 10/9
                    except ValueError:
                        continue

            image_path_to_scores[image_path] = scores_with_raters_of_image

    return image_path_to_scores


def load_rater_to_personalized_scores():
    rater_to_scores = {}
    for image_path, scores in load_image_path_to_personalized_scores().items():
        for rater, score in scores.items():
            if rater not in rater_to_scores:
                rater_to_scores[rater] = {}

            rater_to_scores[rater][image_path] = score

    return rater_to_scores


def test(predictor, image_path_to_score):
    avg_body = []
    for image_path, score in image_path_to_score.items():
        bgr = cv2.imread(image_path)

        human_beauty_score, _ = predictor.predict_physical_beauty(bgr)

        if human_beauty_score is not None:
            avg_body.append(abs(human_beauty_score - score))

    print(np.mean(np.array(avg_body)))


if __name__ == "__main__":
    predictor = VisualAttractiveness()

    test(predictor, load_image_path_to_score())
