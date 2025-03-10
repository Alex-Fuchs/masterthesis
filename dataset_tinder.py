import os
import math
import random
from glob import glob

import cv2

import matplotlib.pyplot as plt

import torch

import visual_attractiveness


def examples(predictor, folder_name, rows=5, cols=10, k=500):
    scores = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

    filenames = [y for x in os.walk(folder_name) for y in glob(os.path.join(x[0], '*.jpg'))]
    random.shuffle(filenames)

    for image_path in filenames[:k]:
        bgr = cv2.imread(image_path)

        human_beauty_score, _ = predictor.predict_physical_beauty(bgr)

        if human_beauty_score is not None:
            scores[math.ceil(human_beauty_score)].append(cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows):
        for j in range(cols):
            axes[i, j].axis('off')

            if len(scores[j + 1]) > i:
                axes[i, j].imshow(scores[j + 1][i])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('tinder_clip_500.png')


if __name__ == "__main__":
    predictor = visual_attractiveness.VisualAttractiveness()

    examples(predictor, '/Users/alexanderfuchs/Desktop/WS_24_25/TinderBotz/data_woman2man/geomatches/images')