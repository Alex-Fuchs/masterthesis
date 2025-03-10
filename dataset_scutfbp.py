import os

import cv2

import torch
import numpy as np

import visual_attractiveness


def load_image_path_to_score(folder_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/Images',
                             file_name='/Users/alexanderfuchs/Desktop/WS_24_25/attractiveness_model/SCUT-FBP5500_v2/train_test_files/All_labels.txt'):
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

    print(f'MAE: {np.mean(np.array(avg_body))}, MSE: {np.mean(np.pow(np.array(avg_body), 2))}')


if __name__ == "__main__":
    text_features = torch.load("weights/text_features_mebeauty_100k.pth")
    text_features = torch.tensor(text_features, dtype=torch.float32)

    predictor = visual_attractiveness.VisualAttractiveness(text_features=text_features)

    test(predictor, load_image_path_to_score())

    # ['attractive', 'unattractive']:                                                                                                       MAE: 1.4721324512290954, MSE: 3.575172415565474
    # ['appealing', 'unappealing']:                                                                                                         MAE: 2.1148055309577782, MSE: 6.233912852647225
    # ['aesthetic', 'unaesthetic']:                                                                                                         MAE: 1.974832114435434, MSE: 5.240584309337174
    # ['good-looking', 'ugly-looking']:                                                                                                     MAE: 1.6096513173856004, MSE: 3.872883618319324
    # ['beautiful, handsome', 'ugly']:                                                                                                      MAE: 1.4359900602034728, MSE: 2.9551007991398315
    # ['appealing, charismatic, charming, seductive', 'ugly, unpleasing, hideous, grotesque']:                                              MAE: 1.7151793189819653, MSE: 3.900068934462793

    # ['Well-lit, confident, and natural with a clear, engaging focus', 'Poor lighting, obscured face, awkward angles, or heavy filters']:  MAE: 2.3638992547869684, MSE: 6.761736798584162
    # ['Bright, clear, and high-quality with a genuine smile', 'Dark, blurry, or low-quality with poor lighting']:                          MAE: 3.7010724281938194, MSE: 19.639175419087476
    # ['Natural, confident, and effortlessly engaging', 'Overly staged, expressionless, or disengaged']:                                    MAE: 2.3222090041326484, MSE: 8.710079304705213
    # ['Well-groomed, well-dressed, and in a flattering setting', 'Messy background, awkward angles, or extreme filters']:                  MAE: 2.2673681362456084, MSE: 6.8290847159904
    # ['Solo, focused, and free of distractions', 'Face obscured by sunglasses, hats, or heavy edits']:                                     MAE: 1.7922986118920645, MSE: 5.095702744805579
    # ['Candid and approachable without excessive editing', 'Too many group photos making it unclear who you are']:                         MAE: 1.638373930533131, MSE: 4.083435850890683

    # fine-tuned: MAE: 1.1782229793095589, MSE: 2.161717152968094
