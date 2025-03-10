import math
import random

from glob import glob

import os

import cv2

import matplotlib.pyplot as plt

import numpy as np
import torch

import visual_attractiveness


def load_image_path_to_latent(predictor, filenames=None, folder_name='./MEBeauty-database-main/original_images'):
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


def load_image_path_to_score(file_name='./MEBeauty-database-main/scores/generic_scores_all.csv'):
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
                score *= 10 / 9
            except ValueError:
                continue

            image_path_to_score[image_path] = score

    return image_path_to_score


def load_image_path_to_personalized_scores(file_name='./MEBeauty-database-main/scores/generic_scores_all.csv'):
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
                        score = score.replace(',', '.')
                        score = float(score)
                        score -= 1
                        score *= 10 / 9
                    except ValueError:
                        continue

                    scores_with_raters_of_image[column_name] = score

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

    print(f'MAE: {np.mean(np.array(avg_body))}, MSE: {np.mean(np.pow(np.array(avg_body), 2))}')


def examples(predictor, image_path_to_score, rows=5, cols=10):
    scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

    image_path_to_score = list(image_path_to_score.items())
    random.shuffle(image_path_to_score)

    for image_path, score in image_path_to_score:
        bgr = cv2.imread(image_path)

        human_beauty_score, _ = predictor.predict_physical_beauty(bgr)

        if human_beauty_score is not None:
            scores[math.ceil(human_beauty_score)].append(cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR))

            if all([len(l) >= rows for _, l in scores.items()]):
                break

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows):
        for j in range(cols):
            axes[i, j].axis('off')

            if len(scores[j + 1]) > i:
                axes[i, j].imshow(scores[j + 1][i])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('mebeauty_clip.png')


if __name__ == "__main__":
    text_features = torch.load("weights/text_features_mebeauty_100k.pth")
    text_features = torch.tensor(text_features, dtype=torch.float32)

    predictor = visual_attractiveness.VisualAttractiveness(captions=['Candid and approachable without excessive editing', 'Too many group photos making it unclear who you are'])

    test(predictor, load_image_path_to_score())

    # ['attractive', 'unattractive']:                                                                                                       MAE: 1.3908634361130359, MSE: 3.0459226081473085
    # ['appealing', 'unappealing']:                                                                                                         MAE: 1.2645126415587038, MSE: 2.5127056552245874
    # ['aesthetic', 'unaesthetic']:                                                                                                         MAE: 1.4491079707331704, MSE: 3.2599016934045593
    # ['good-looking', 'ugly-looking']:                                                                                                     MAE: 1.6096513173856004, MSE: 3.872883618319324
    # ['beautiful, handsome', 'ugly']:                                                                                                      MAE: 3.044576662643919, MSE: 11.725727413105918
    # ['appealing, charismatic, charming, seductive', 'ugly, unpleasing, hideous, grotesque']:                                              MAE: 2.118117987207299, MSE: 6.472258001753557

    # ['Well-lit, confident, and natural with a clear, engaging focus', 'Poor lighting, obscured face, awkward angles, or heavy filters']:  MAE: 3.4309821218245835, MSE: 14.226367948725079
    # ['Bright, clear, and high-quality with a genuine smile', 'Dark, blurry, or low-quality with poor lighting']:                          MAE: 3.7436524051060167, MSE: 18.125032716886906
    # ['Natural, confident, and effortlessly engaging', 'Overly staged, expressionless, or disengaged']:                                    MAE: 2.3664969156412665, MSE: 7.797916525529965
    # ['Well-groomed, well-dressed, and in a flattering setting', 'Messy background, awkward angles, or extreme filters']:                  MAE: 3.6007936988460956, MSE: 16.985701985502537
    # ['Solo, focused, and free of distractions', 'Face obscured by sunglasses, hats, or heavy edits']:                                     MAE: 3.4141223426047875, MSE: 14.239243808108224
    # ['Candid and approachable without excessive editing', 'Too many group photos making it unclear who you are']:                         MAE: 3.485775585107037, MSE: 14.576504192435257

    # fine-tuned: MAE: 0.6471963986818654, MSE: 0.6707251640905796
